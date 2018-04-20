# Simple experiment
require(randomForest)

getData <- function(id) {
	d1=read.table("student-mat.csv",sep=",",header=TRUE)
	d2=read.table("student-por.csv",sep=",",header=TRUE)
	if (id == 1) {
		dataset <- d1
	} else {
		dataset <- d2
	}

	dataset

	## use code below for experimenting with merged tables
	## keep in mind that G1, G2, G3 are different in both tables
	# dataset <- rbind(d1, d2)
	# remove duplicate rows (basing on some collumns)
	# oldNrow = nrow(dataset)
	# duplicateCriteria = c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet")
	# dataset <- dataset[!duplicated(dataset[duplicateCriteria]),]
	# nrow(dataset)
	# stopifnot(oldNrow - nrow(dataset) == 382) # as written in description on kaggle
}

# convert 2 ordinal attributes (Walc, Dalc) to 1 binary (alc)
walc_dalc_to_alc <- function(dataset) {
    # visualize 2 attributes
    library(ggplot2)
    qplot(dataset$Walc, dataset$Dalc, geom='bin2d')
    # TODO (optional) : find better visualization

    # TODO
}

doExperiment <- function(dataset) {
	# separate a test set from dataset
	smp_size <- floor(0.75 * nrow(dataset))
	set.seed(123)
	# R doesn't have unpack syntax, so a nice oneliner is probably not possible
	train_ind <- sample(seq_len(nrow(dataset)), size = smp_size)
	train <- dataset[train_ind, ]
	test <- dataset[-train_ind, ]

	param <- Dalc ~ (school+sex+age+address+famsize+Pstatus+Medu+Fedu+Mjob+Fjob
					+reason+guardian+traveltime+studytime+failures+schoolsup+famsup
					+paid+activities+nursery+higher+internet+romantic+famrel+freetime
					+goout+health+absences+G1+G2+G3)
	fit <- randomForest(param, data=train, importance=TRUE, ntree=2000)

	# results
	varImpPlot(fit)
	predicted = predict(fit, test)
	expected = test$Dalc
	mean(abs(predicted - expected))
}

# doExperiment(getData(1))
# doExperiment(getData(2))
walc_dalc_to_alc(getData(1))
