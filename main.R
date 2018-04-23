# Simple experiment
require(randomForest)
require(ggplot2)
require(cluster)
require(ade4)

library(rpart)
library(rattle)
library(rpart.plot)

joinDataframes <- function(listOfDf) {
    df <- do.call("rbind", listOfDf)
    df
}

# returns dataset (one of 2 possible)
getData <- function(id) {
    d1=read.table("student-mat.csv",sep=",",header=TRUE)
    d2=read.table("student-por.csv",sep=",",header=TRUE)
    if (id == 1) {
        dataset <- d1
    } else {
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

# convert 2 ordinal attributes (Walc, Dalc) to 1 binary (alc)
walcToDalc <- function(dataset) {
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
trainSingleTreeClassifier <- function(dataset, draw=FALSE) {
    param <- Drink ~ (school+sex+age+address+famsize+Pstatus+Medu+Fedu+Mjob+Fjob
                    +reason+guardian+traveltime+studytime+failures+schoolsup+famsup
                    +paid+activities+nursery+higher+internet+romantic+famrel+freetime
                    +goout+health+absences+G1+G2+G3)
    # tr <- rpart(param,data = dataset,method="class",control =rpart.control(minsplit = 30,minbucket=12, cp=0.005,maxdepth=10))
    tr <- rpart(param,data = dataset,method="class",control =rpart.control(minsplit = 30,minbucket=12, cp=0.005,maxdepth=10))
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
trainRandomForestClssifier <- function(trainset, draw=FALSE) {
    param <- Drink ~ (school+sex+age+address+famsize+Pstatus+Medu+Fedu+Mjob+Fjob
                    +reason+guardian+traveltime+studytime+failures+schoolsup+famsup
                    +paid+activities+nursery+higher+internet+romantic+famrel+freetime
                    +goout+health+absences+G1+G2+G3)
    fit <- randomForest(param, data=trainset, importance=TRUE, ntree=200)
    if (draw) {
        varImpPlot(fit)
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

doExperiment <- function(dataset, trainFun) {
    set.seed(123)
    dataset <- dataset[sample(nrow(dataset)),]
    errFun <- function(predFun, testset) {
        getClassifierError(predFun, testset, "Drink")
    }
    crossValidation(dataset, trainFun, errFun, 8)
}

main <- function() {
    # only draws plots
    calcClustering(mergedData(getData(1), getData(2)))

    # draw plots for classifiers done on whole data
    for (i in 1:2) {
        dataset <- walcToDalc(getData(i))
        trainSingleTreeClassifier(dataset, draw=T)
        trainRandomForestClssifier(dataset, draw=T)
    }

    for (i in 1:2) {
        if (i==1) {
            print("---------------Math")
        } else {
            print("---------------Portugese")
        }
        dataset <- walcToDalc(getData(i))
        print("randomForest error:")
        print(doExperiment(dataset, trainRandomForestClssifier))
        print("decTree error:")
        print(doExperiment(dataset, trainSingleTreeClassifier))
    }
}

main()
