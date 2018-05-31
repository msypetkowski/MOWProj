# Simple experiment
require(randomForest)
library(ade4)
library(rpart)
library(rattle)
library(ggplot2)
library(xtable)
library(Hmisc)

joinDataframes <- function(listOfDf) {
    df <- do.call("rbind", listOfDf)
    df
}

# returns dataset (one of 2 possible)
getData <- function(id) {
    if (id == 1) {
        d=read.table("student-mat.csv",sep=",",header=TRUE)
        names(d)[names(d) == 'G1'] <- 'M_G1'
        names(d)[names(d) == 'G2'] <- 'M_G2'
        names(d)[names(d) == 'G3'] <- 'M_G3'
        names(d)[names(d) == 'paid'] <- 'M_paid'
    } else {
        d=read.table("student-por.csv",sep=",",header=TRUE)
        names(d)[names(d) == 'G1'] <- 'P_G1'
        names(d)[names(d) == 'G2'] <- 'P_G2'
        names(d)[names(d) == 'G3'] <- 'P_G3'
        names(d)[names(d) == 'paid'] <- 'P_paid'
    }
    d
}

modefunc <- function(x){
    tabresult <- tabulate(x)
    themode <- which(tabresult == max(tabresult))
    return(levels(x)[themode])
}

# returns merged dataset
mergedData <- function(d1, d2) {
    print("Total math and portugese students datasets columns count:")
    print(length(names(d1)))
    print(length(names(d2)))

    duplicateCriteria = c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet")
    additional1 = c("M_G1", "M_G2", "M_G3", "M_paid")
    additional2 = c("P_G1", "P_G2", "P_G3", "P_paid")
    dataset1 <- merge(d1, d2[, c(duplicateCriteria, additional2)], by=duplicateCriteria, all.x=TRUE)
    dataset2 <- merge(d2, d1[, c(duplicateCriteria, additional1)], by=duplicateCriteria, all.x=TRUE)
    dataset <- rbind(dataset1, dataset2)
    dataset <- dataset[!duplicated(dataset[duplicateCriteria]),]

    print("Total math and students:")
    print(nrow(d1))
    print(nrow(d2))
    print("Total new dataset rows:")
    print(nrow(dataset))
    print("Duplicate students count:")
    print(nrow(d1) + nrow(d2) - nrow(dataset))
    print("New dataset columns count:")
    print(length(names(dataset)))
    print("New dataset columns:")
    print(names(dataset))

    stopifnot(nrow(d1) + nrow(d2) - nrow(dataset) == 382) # as written in description on kaggle

    # replace <NA> values with mean or mode values
    for(name in c(additional1, additional2)) {
        column <- dataset[,name]
        lev <- levels(dataset[,name])
        if (!is.null(lev)) {
            # nominal
            mo <- modefunc(column[!is.na(column)])
            dataset[is.na(column), name] <- mo
        } else {
            # numeric
            dataset[is.na(column), name] <- mean(column, na.rm = TRUE)
        }
    }
    dataset
}

# find criteruim for converting 2 ordinal attributes (Walc, Dalc) to 1 binary (alc)
calcClustering <- function(dataset) {
    # visualize 2 attributes
    # TODO (optional) : find better visualization
    print(qplot(dataset$Walc, dataset$Dalc, geom='bin2d'))

    # do clustering - increased Dalc importance
    df <- data.frame(scale(dataset$Walc), scale(dataset$Dalc))
    plot(df)
    km <- kmeans(df, centers=rbind(c(-1,-1), c(1,1)))
    kmeansRes<-factor(km$cluster)
    s.class(df,fac=kmeansRes, add.plot=TRUE, col=rainbow(nlevels(kmeansRes)))
}

# convert 2 ordinal attributes (Walc, Dalc) to 1 binary (Drink)
walcDalcToDrink <- function(dataset) {
    d1 = dplyr::filter(dataset, 
        (Dalc <= 2 & Walc <= 1) |
        (Dalc <= 1 & Walc <= 2))
    print("Non-drinking studens ratio")
    print(nrow(d1)/nrow(dataset))

    d2 = dplyr::filter(dataset, !(
        (Dalc <= 2 & Walc <= 1) |
        (Dalc <= 1 & Walc <= 2)))
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

getParam <- function() {
    param <- Drink ~ (school+sex+age+address+famsize+Pstatus+Medu+Fedu+Mjob+Fjob
                    +reason+guardian+traveltime+studytime+failures+schoolsup+famsup
                    +M_paid+P_paid+activities+nursery+higher+internet+romantic+famrel+freetime+goout+health+absences
                    +M_G1+M_G2+M_G3+P_G1+P_G2+P_G3)
    # param <- Drink ~ (sex+Fjob+higher+famsize+reason
    #                +goout+P_G1+P_G2+P_G3+M_G1+M_G2+M_G3+studytime+absences
    #                 +freetime+famrel+health+age+M_G1+nursery+M_paid+Mjob)
    param
}

# returns predict function with 1 param - testset
trainSingleTreeClassifier <- function(dataset, draw=FALSE, cp=0.01, minbucket=15, maxdepth=15, split="gini") {
    param <- getParam()
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
trainRandomForestClssifier <- function(trainset, draw=FALSE, ntree=500, nodesize=10, mtry=6, maxnodes=30) {
    param <- getParam()
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

    errSamples <- c()
    for (i in 1:n) {
        train <- joinDataframes(parts[-i])
        test <- parts[[i]]
        stopifnot(nrow(train) + nrow(test) == nrow(dataset))
        fit <- trainFunction(train)
        e <- errorFunction(fit, test)
        errSamples <- c(errSamples, c(e))
    }
    errSamples
}

getClassifierError <- function(predFun, testset, attrName) {
    predicted <- predFun(testset)
    stopifnot(length(predicted) == nrow(testset))
    sum(predicted!= testset[[attrName]]) / nrow(testset)
}

doExperiment <- function(dataset, trainFun, repTimes=10) {
    errSamples <- vector()
    set.seed(123)
    for (i in 1:repTimes) {
        dataset <- dataset[sample(nrow(dataset)),]
        errFun <- function(predFun, testset) {
            getClassifierError(predFun, testset, "Drink")
        }
        errSamples <- c(errSamples, crossValidation(dataset, trainFun, errFun, 8))
    }
    errSamples
}

# train many decision trees
decTreeTest <- function(dataset1) {
    paramCp = list(0.0001, 0.001, 0.01, 0.1, 0.3)
    paramMinBucket = list(5, 15)
    paramMaxDepth = list(5, 15)
    paramSplit = list("gini", "information")
    records <- apply(expand.grid(paramCp, paramMinBucket, paramMaxDepth, paramSplit), 1, FUN = function(x) {
        samples <- doExperiment(dataset1, function(ds) {
                    trainSingleTreeClassifier(ds, cp=x[[1]], minbucket=x[[2]], maxdepth=x[[3]], split=x[[4]])
        })
        data.frame(
                    c(x[[1]]),
                    c(x[[2]]),
                    c(x[[3]]),
                    c(x[[4]]),
                    c(mean(samples)),
                    c(sd(samples)),
                    c(min(samples)),
                    c(max(samples))
        )
    })
    records <- joinDataframes(records)
    records <- setNames(records, c("cp", "minbucket", "maxdepth", "split", "err_mean", "err_sd", "err_min", "err_max"))
    records <- records[with(records, order(err_mean)), ]
    print(records)
    print(xtable(records, type = "latex", digits=c(0, 5, 2, 2, 0, 2, 2, 2, 2),
                 label = "table:singleResults", caption = "Various single decision trees results (sorted by mean error)"),
          file = "singleResults.tex", caption.placement = "top", include.rownames=FALSE)
}

# train many random forests
forestTest <- function(dataset1) {
    paramNtree = list(20, 60, 500)
    paramNodesize = list(1, 10)
    paramMtry = list(6, 10, 35)
    paramMaxnodes = list(8, 30, 500)
    records <- apply(expand.grid(paramNtree, paramNodesize, paramMtry, paramMaxnodes), 1, FUN = function(x) {
        samples <- doExperiment(dataset1, function(ds) {
                    trainRandomForestClssifier(ds, ntree=x[[1]], nodesize=x[[2]], mtry=x[[3]], maxnodes=x[[4]])
        })
        data.frame(
                    c(x[[1]]),
                    c(x[[2]]),
                    c(x[[3]]),
                    c(x[[4]]),
                    c(mean(samples)),
                    c(sd(samples)),
                    c(min(samples)),
                    c(max(samples))
        )
    })
    records <- joinDataframes(records)
    records <- setNames(records, c("ntree", "nodesize", "mtry", "maxnodes", "err_mean", "err_sd", "err_min", "err_max"))
    records <- records[with(records, order(err_mean)), ]
    print(records)
    print(xtable(records, type = "latex",
                 label="table:forestResults", caption="Various random forests results (sorted by mean error)"),
          file = "forestResults.tex", caption.placement = "top", tabular.environment = "longtable", include.rownames=FALSE)
}

# train many random forests with various mtry value (and draw plot)
forestTestDetailedMtry <- function(dataset1) {
    paramMtry = 1:20 * 2
    paramMtry = t(paramMtry[paramMtry <= 35])
    # paramMtry = cbind(1, 3, 5, 8, 10, 20, 25, 35)
    records <- apply(paramMtry, 2, FUN = function(x) {
        samples <- doExperiment(dataset1, function(ds) {
                    trainRandomForestClssifier(ds, mtry=x[[1]])
        })
        data.frame(
                    c(x[[1]]),
                    c(mean(samples)),
                    c(sd(samples)),
                    c(min(samples)),
                    c(max(samples))
        )
    })
    records <- joinDataframes(records)
    records <- setNames(records, c("mtry", "err_mean", "err_sd", "err_min", "err_max"))
    # records <- records[with(records, order(err_mean)), ]
    print(records)
    print(xtable(records, type = "latex",
                 label="table:forestResults2", caption="Random forest results for various mtry parameter values"),
          file = "forestResults2.tex", caption.placement = "top", include.rownames=FALSE)
    # records <- records[with(records, order(mtry)), ]
    plot(records$mtry, records$err_mean, type="l")

    plot(records$mtry, records$err_mean, type="n")
        with (
            data = records,
            expr = errbar(mtry, err_mean, err_mean+err_sd, err_mean-err_sd) # , add=T) # , pch=1, cap=.1)
        )
}

# train many random forests with various maxnodes value (and draw plot)
forestTestDetailedMaxnodes <- function(dataset1) {
    paramMaxnodes = t(1:20 * 5)
    records <- apply(paramMaxnodes, 2, FUN = function(x) {
        samples <- doExperiment(dataset1, function(ds) {
                    trainRandomForestClssifier(ds, maxnodes=x[[1]])
        })
        data.frame(
                    c(x[[1]]),
                    c(mean(samples)),
                    c(sd(samples)),
                    c(min(samples)),
                    c(max(samples))
        )
    })
    records <- joinDataframes(records)
    records <- setNames(records, c("maxnodes", "err_mean", "err_sd", "err_min", "err_max"))
    print(records)
    print(xtable(records, type = "latex",
                 label="table:forestResults2", caption="Random forest results for various maxnodes parameter values"),
          file = "forestResults2.tex", caption.placement = "top", include.rownames=FALSE)
    plot(records$maxnodes, records$err_mean, type="l")

    plot(records$maxnodes, records$err_mean, type="n")
        with (
            data = records,
            expr = errbar(maxnodes, err_mean, err_mean+err_sd, err_mean-err_sd)
        )
}
binaryEntropy <- function(ratio) {
    -(ratio * log(ratio + 1e-100) + (1-ratio) * log((1-ratio) + 1e-100))
}

# train many random forests with various nodesize value (and draw plot)
forestTestDetailedNodesize <- function(dataset1) {
    paramNodesize = t(c(1, 1 + 1:10 * 3))
    records <- apply(paramNodesize, 2, FUN = function(x) {
        samples <- doExperiment(dataset1, function(ds) {
                    trainRandomForestClssifier(ds, nodesize=x[[1]])
        })
        data.frame(
                    c(x[[1]]),
                    c(mean(samples)),
                    c(sd(samples)),
                    c(min(samples)),
                    c(max(samples))
        )
    })
    records <- joinDataframes(records)
    records <- setNames(records, c("nodesize", "err_mean", "err_sd", "err_min", "err_max"))
    print(records)
    print(xtable(records, type = "latex",
                 label="table:forestResults2", caption="Random forest results for various nodesize parameter values"),
          file = "forestResults2.tex", caption.placement = "top", include.rownames=FALSE)
    plot(records$nodesize, records$err_mean, type="l")

    plot(records$nodesize, records$err_mean, type="n")
        with (
            data = records,
            expr = errbar(nodesize, err_mean, err_mean+err_sd, err_mean-err_sd)
        )
}
binaryEntropy <- function(ratio) {
    -(ratio * log(ratio + 1e-100) + (1-ratio) * log((1-ratio) + 1e-100))
}

measureSingleSplits <- function(dataset) {
    records <- list()
    ttestRecords <- list()
    for (name in names(dataset)) {
        if (name=="Drink") {
            next
        }
        lev <- levels(dataset[,name])

        # entropy for nominal attributes
        # and t-test for numeric
        if (!is.null(lev)) {
            names <- c(name, names)
            weights <- c()
            entropies <- c()
            for (l in lev) {
                mask <- dataset[,name] == l
                d <- dataset[mask, "Drink"]
                total <- length(d)
                drink <- sum(d == 2)
                ratio <- drink/total
                weights <- c(weights, total)
                entropies <- c(entropies, binaryEntropy(ratio))
            }
            afterSplitEntropy <- sum(entropies*weights) / sum(weights)
            d <- dataset[, "Drink"]
            total <- length(d)
            drink <- sum(d == 2)
                ratio <- drink/total
            beforeSplitEntropy <- binaryEntropy(ratio)
            informationGain <- beforeSplitEntropy - afterSplitEntropy
            records[[length(records) + 1L]] <- data.frame(name, length(lev), informationGain)
        } else {
            print(name)
            res = t.test(dataset[dataset$Drink=="1", name],
                         dataset[dataset$Drink=="2", name])
            ttestRecords[[length(ttestRecords) + 1L]] <- data.frame(name, res$p.value)
        }
    }
    # print(records)
    records <- joinDataframes(as.list(records))
    records <- setNames(records, c("name", "levels", "informationGain"))
    records <- records[with(records, order(informationGain)), ]
    print(records)
    print(xtable(records, type = "latex", digits=c(0, 0, 0, 7),
                 label="table:nominalIG", caption="Information gain for single value based splits for nominal attributes"),
          file = "nominalIG.tex", caption.placement = "top", table.placement = "H", include.rownames=FALSE)
    barplot(records[,"informationGain"] , horiz=TRUE, names.arg=records[,"name"], las=1, cex.names=0.8)
    records

    ttestRecords <- joinDataframes(as.list(ttestRecords))
    ttestRecords <- setNames(ttestRecords, c("name", "p.value"))
    ttestRecords <- ttestRecords[with(ttestRecords, order(-p.value)), ]
    barplot(ttestRecords[,"p.value"] , horiz=TRUE, names.arg=ttestRecords[,"name"], las=1, cex.names=0.8)
}

main <- function() {
    set.seed(123)
    data <- mergedData(getData(1), getData(2))

    # only draws plots
    calcClustering(data)

    # get dataset
    dataset1 = walcDalcToDrink(data)

    # draw plots
    trainSingleTreeClassifier(dataset1, draw=T)
    trainRandomForestClssifier(dataset1, draw=T)

    # measure importance of single attributes
    measureSingleSplits(dataset1);

    print("-----------single dec tree experiments")
    decTreeTest(dataset1)
    print("-----------random forest experiments")
    forestTest(dataset1)

    print("-----------random forest detailed experiments with mtry parameter")
    forestTestDetailedMtry(dataset1)

    print("-----------random forest detailed experiments with maxnodes parameter")
    forestTestDetailedMaxnodes(dataset1)

    print("-----------random forest detailed experiments with nodesize parameter")
    forestTestDetailedNodesize(dataset1)
}

main()
