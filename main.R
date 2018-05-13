# Simple experiment
require(randomForest)
# require(ggplot2)
# require(cluster)
library(ade4)
library(rpart)
library(rattle)
# require(rpart.plot)
library(ggplot2)

joinDataframes <- function(listOfDf) {
    df <- do.call("rbind", listOfDf)
    df
}

# returns dataset (one of 2 possible)
getData <- function(id) {
    d1=read.table("student-mat.csv",sep=",",header=TRUE)
    d2=read.table("student-por.csv",sep=",",header=TRUE)
    if (id == 1) {
        # names(d1)[names(d1) == 'G1'] <- 'M_G1'
        # names(d1)[names(d1) == 'G2'] <- 'M_G2'
        # names(d1)[names(d1) == 'G3'] <- 'M_G3'
        # d1$P_G1 = NA
        # d1$P_G2 = NA
        # d1$P_G3 = NA
        dataset <- d1
    } else {
        # names(d2)[names(d2) == 'G1'] <- 'P_G1'
        # names(d2)[names(d2) == 'G2'] <- 'P_G2'
        # names(d2)[names(d2) == 'G3'] <- 'P_G3'
        # d2$M_G1 = NA
        # d2$M_G2 = NA
        # d2$M_G3 = NA
        dataset <- d2
    }
    dataset
}

# returns merged dataset
mergedData <- function(d1, d2) {
    # G1, G2, G3 are different in both tables
    dataset <- rbind(d1, d2)
    # remove duplicate rows (basing on some collumns)
    oldNrow = nrow(dataset)
    duplicateCriteria = c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet")
    dataset <- dataset[!duplicated(dataset[duplicateCriteria]),]
    nrow(dataset)
    stopifnot(oldNrow - nrow(dataset) == 382) # as written in description on kaggle
    dataset$goout[5:50] = NA
    dataset
}

# find criteruim for converting 2 ordinal attributes (Walc, Dalc) to 1 binary (alc)
calcClustering <- function(dataset) {
    # visualize 2 attributes
    # TODO (optional) : find better visualization
    print(qplot(dataset$Walc, dataset$Dalc, geom='bin2d'))

    # do clustering
    df <- data.frame(dataset$Walc, dataset$Dalc)
    plot(df)
    # class 1 will mean non-drinking, 2 - drinking
    km <- kmeans(df, centers=rbind(c(1,1), c(3,3)))
    kmeansRes<-factor(km$cluster)
    s.class(df,fac=kmeansRes, add.plot=TRUE, col=rainbow(nlevels(kmeansRes)))

    # do clustering - increased Dalc importance
    df <- data.frame(dataset$Walc, dataset$Dalc * 1.5)
    plot(df)
    km <- kmeans(df, centers=rbind(c(1,1), c(3,3)))
    kmeansRes<-factor(km$cluster)
    s.class(df,fac=kmeansRes, add.plot=TRUE, col=rainbow(nlevels(kmeansRes)))

    # ret <- dataset[ , -which(names(dataset) %in% c("Walc","Dalc"))]
    # ret$Drink <- km$cluster
    # ret
}

# convert 2 ordinal attributes (Walc, Dalc) to 1 binary (Drink)
walcDalcToDrink <- function(dataset) {
    d1 = dplyr::filter(dataset, 
        (Dalc <= 2 & Walc <= 2) |
        (Dalc <= 3 & Walc <= 1) |
        (Dalc <= 1 & Walc <= 3))
    print("Non-drinking studens ratio")
    print(nrow(d1)/nrow(dataset))

    d2 = dplyr::filter(dataset, !(
        (Dalc <= 2 & Walc <= 2) |
        (Dalc <= 3 & Walc <= 1) |
        (Dalc <= 1 & Walc <= 3)))
    print("Drinking studens ratio")
    print(nrow(d2)/nrow(dataset))

    stopifnot(nrow(d1) + nrow(d2) == nrow(dataset))

    d1$Drink = 1
    d2$Drink = 2

    ret <- rbind(d1, d2)
    ret <- ret[ , -which(names(ret) %in% c("Walc","Dalc"))]
    ret$Drink <- as.factor(ret$Drink) # we want classification
    stopifnot(nrow(ret) == nrow(dataset))
    stopifnot(ncol(ret) == ncol(dataset) - 1)
    ret
}

# separate a random test set from dataset
randomDivideDataset <- function(dataset, trainRatio) {
    smp_size <- floor(trainRatio * nrow(dataset))
    set.seed(123)
    # R doesn't have unpack syntax, so a nice oneliner is probably not possible
    train_ind <- sample(seq_len(nrow(dataset)), size = smp_size)
    train <- dataset[train_ind, ]
    test <- dataset[-train_ind, ]
    stopifnot(nrow(train) + nrow(test) == nrow(dataset))
    list("train" = train, "test" = test)
}

# returns predict function with 1 param - testset
trainSingleTreeClassifier <- function(dataset, draw=FALSE, cp=0.001, minbucket=5, maxdepth=5, split="gini") {
    param <- Drink ~ (school+sex+age+address+famsize+Pstatus+Medu+Fedu+Mjob+Fjob
                    +reason+guardian+traveltime+studytime+failures+schoolsup+famsup
                    +paid+activities+nursery+higher+internet+romantic+famrel+freetime +goout+health+absences
                    +G1+G2+G3) # +M_G1+M_G2+M_G3+P_G1+P_G2+P_G3)
    # tr <- rpart(param,data = dataset,method="class",control =rpart.control(minsplit = 30,minbucket=12, cp=0.005,maxdepth=10))
    tr <- rpart(param,data = dataset,method="class",control = rpart.control(
                                    cp=cp,
                                    minsplit = minbucket*2,
                                    minbucket= minbucket,
                                    maxdepth=maxdepth
                             ), parms=list(split=split))
    if (draw) {
        fancyRpartPlot(tr)
    }
    predFun <- function(testset) {
        p <- predict(tr, testset)
        as.integer(p[, 1] < 0.5) + 1
    }
    predFun
}

# returns predict function with 1 param - testset

trainRandomForestClssifier <- function(trainset, draw=FALSE, ntree=500, nodesize=10, seed=2, mtry=31, maxnodes=5) {
    set.seed(seed)
    param <- Drink ~ (school+sex+age+address+famsize+Pstatus+Medu+Fedu+Mjob+Fjob
                    +reason+guardian+traveltime+studytime+failures+schoolsup+famsup
                    +paid+activities+nursery+higher+internet+romantic+famrel+freetime
                    +paid+activities+nursery+higher+internet+romantic+famrel+freetime +goout+health+absences
                    +G1+G2+G3) # +M_G1+M_G2+M_G3+P_G1+P_G2+P_G3)
    fit <- randomForest(param, data=trainset, importance=draw, ntree=ntree, nodesize=nodesize, mtry=mtry, maxnodes=maxnodes)
    if (draw) {
        varImpPlot(fit)
        # plot(fit)
        # print(getTree(fit, k=1, labelVar=F))
    }
    predFun <- function(testset) {
        predict(fit, testset)
    }
    predFun
}

crossValidation <- function(dataset, trainFunction, errorFunction, partitionsCount) {
    n <- partitionsCount
    nr <- nrow(dataset)
    # TODO: some samples can be lost
    f <- rep(1:ceiling(n), each=floor(nr/n), length.out=nr)
    parts <- split(dataset, f)
    stopifnot(nrow(dataset) - nrow(joinDataframes(parts)) >= 0)
    stopifnot(nrow(dataset) - nrow(joinDataframes(parts)) < nr/n)

    errSum <- 0.0
    for (i in 1:n) {
        train <- joinDataframes(parts[-i])
        test <- parts[[i]]
        stopifnot(nrow(train) + nrow(test) == nrow(dataset))
        fit <- trainFunction(train)
        e <- errorFunction(fit, test)
        errSum <- errSum + e
    }
    errSum / n
}

getClassifierError <- function(predFun, testset, attrName) {
    predicted <- predFun(testset)
    stopifnot(length(predicted) == nrow(testset))
    sum(predicted!= testset[[attrName]]) / nrow(testset)
}

doExperiment <- function(dataset, trainFun, repTimes=5) {
    errSum <- 0.0
    for (i in 1:repTimes) {
        set.seed(123 + i)  # more elegant way would be using random generator object
        dataset <- dataset[sample(nrow(dataset)),]
        errFun <- function(predFun, testset) {
            getClassifierError(predFun, testset, "Drink")
        }
        errSum <- errSum + crossValidation(dataset, trainFun, errFun, 8)
    }
    errSum / repTimes
}

decTreeTest <- function(dataset1) {
    # train many dec trees
    paramCp = list(0.0001, 0.001, 0.01, 0.1, 0.3)
    paramMinBucket = list(5, 10, 15)
    paramMaxDepth = list(5, 15)
    paramSplit = list("gini", "information")
    records <- apply(expand.grid(paramCp, paramMinBucket, paramMaxDepth, paramSplit), 1, FUN = function(x) {
        error <- doExperiment(dataset1, function(ds) {
                    trainSingleTreeClassifier(ds, cp=x[[1]], minbucket=x[[2]], maxdepth=x[[3]], split=x[[4]])
        })
        data.frame(
                    c(x[[1]]),
                    c(x[[2]]),
                    c(x[[3]]),
                    c(x[[4]]),
                    c(error)
        )
    })
    records <- joinDataframes(records)
    records <- setNames(records, c("cp", "minbucket", "maxdepth", "split", "error"))
    records <- records[with(records, order(error)), ]
    print(records)
}

forestTest <- function(dataset1) {
    # train many random forests
    paramNtree = list(1, 5, 50, 500)
    paramNodesize = list(1, 10)
    # paramSeed = list(1, 2, 3, 4, 5)
    paramMtry = list(1, 6, 10, 31)
    paramMaxnodes = list(5, 10, 30)
    records <- apply(expand.grid(paramNtree, paramNodesize, paramMtry, paramMaxnodes), 1, FUN = function(x) {
        error <- doExperiment(dataset1, function(ds) {
                    trainRandomForestClssifier(ds, ntree=x[[1]], nodesize=x[[2]], mtry=x[[3]], maxnodes=x[[4]])
        })
        data.frame(
                    c(x[[1]]),
                    c(x[[2]]),
                    c(x[[3]]),
                    c(x[[4]]),
                    c(error)
        )
    })
    records <- joinDataframes(records)
    records <- setNames(records, c("ntree", "nodesize", "mtry", "maxnodes", "error"))
    records <- records[with(records, order(error)), ]
    print(records)
}

main <- function() {
    # only draws plots
    calcClustering(mergedData(getData(1), getData(2)))

    # get dataset
    dataset1 = walcDalcToDrink(mergedData(getData(1), getData(2)))
    print(nrow(dataset1))
    print(dataset1[1:20,])

    # # draw plots for classifiers done on whole data
    trainSingleTreeClassifier(dataset1, draw=T)
    trainRandomForestClssifier(dataset1, draw=T)

    # print("-----------single dec tree experiments")
    # # decTreeTest(dataset1)
    # print("-----------random forest experiments")
    # forestTest(dataset1)
}

main()
