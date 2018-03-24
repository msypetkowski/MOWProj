# Proof of Concept
require(randomForest)

d1=read.table("student-mat.csv",sep=",",header=TRUE)
d2=read.table("student-por.csv",sep=",",header=TRUE)
dataset <- rbind(d1, d2)

# remove duplicate rows (basing on some collumns)
oldNrow = nrow(dataset)
duplicateCriteria = c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet")
dataset <- dataset[!duplicated(dataset[duplicateCriteria]),]
nrow(dataset)
stopifnot(oldNrow - nrow(dataset) == 382) # as written in description on kaggle

# separate a test set
# 75% of the sample size
smp_size <- floor(0.75 * nrow(dataset))
set.seed(123)
# R doesn't have unpack syntax, so a nice oneliner is probably not possible
train_ind <- sample(seq_len(nrow(dataset)), size = smp_size)
train <- dataset[train_ind, ]
test <- dataset[-train_ind, ]

param <- Dalc ~ school + sex + age
fit <- randomForest(param, data=train, importance=TRUE, ntree=2000)

# TODO: test/measure
