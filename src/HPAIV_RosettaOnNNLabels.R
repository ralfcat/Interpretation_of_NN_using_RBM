# Flush r-studio and set working directory 
rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
set.seed(0)

# load necessary packages
library(dplyr)
library(R.ROSETTA)

# ======================= Build RBM on NN output ==============================

# load data
dat <- read.csv("../data/NS1/NewTrain2_python.csv", header = F, colClasses = "character")

# change names of truth and prediction
names(dat)[dim(dat)[2] - 1] = "True"
names(dat)[dim(dat)[2]] = "Prediction"

# extract true labels
truth <- dat[,(dim(dat)[2] - 1)]

# remove truth label from data frame
df <- dat[,c(1:(dim(dat)[2] - 2), dim(dat)[2])]

# create reversed data frame and data frame containing positions 85-89
rev_df <- dat[,c((dim(dat)[2] - 2):1, dim(dat)[2])]
centre_df <- dat[,c(85:89, dim(dat)[2])]

# run Rosetta on all three data frame
ros <- rosetta(df, discrete = T, underSample = T, reducer = "Johnson")
rev_ros <- rosetta(rev_df, discrete = T, underSample = T, reducer = "Johnson")
centre_ros <- rosetta(centre_df, discrete = T, underSample = T, reducer = "Genetic")

# combine rules and recalculate their quality measures
comb <- data.frame(rbind(ros$main, rev_ros$main, centre_ros$main))
rec <- recalculateRules(df, comb, discrete = T)
rec <- distinct(rec[rec$pValue<=0.05,])

# check rules
viewRules(rec)

# ======================== Test RBM on NN output ==============================

# load test set
test <- read.csv("../data/NS1/NewTest.csv", colClasses = "character", header = F)

# extract data and NN label
test_truth <- test[,(dim(test)[2] - 1)]
test_df <- test[,c(1:(dim(test)[2] - 2), dim(test)[2])]

names(test_df)[dim(test_df)[2]] <- "Prediction"

# predict classes for test data
pred <- predictClass(test_df[,1:(dim(test_df)[2] - 1)], rec, discrete = T, validate = T, defClass = test_df[,dim(test_df)[2]], normalizeMethod = "rulnum")
pred$accuracy
table(pred$out[,c("currentClass", "predictedClass")])


# ================= Analysis of wrongly classified objects ====================

# extract wrongly classified objects
wrongObj <- which(df[,dim(df)[2]] != truth)

# get supportSet as list
supportSet <- lapply(rec$supportSetRHS, function(x){as.numeric(unlist(strsplit(x, ",")))})

# create heatmap to identify misclassified objects in support set of rules
rules <- list()
for(i in wrongObj){
  tmp = which(unlist(lapply(supportSet, function(x){(i %in% x)})))
  rules[[length(rules) + 1]] <- as.numeric(row.names(rec[tmp,]))
}

r <- unlist(rules) %>% unique

x = lapply(rules, function(x){as.numeric(r %in% x)}) %>% data.frame
names(x) = wrongObj
row.names(x) <- r

heatmap(as.matrix(x), scale = "none", Colv = F, col = c("white", "black"),
        xlab = "Object Number", ylab = "Rule Rank")

# based on heatmap inspect rules 
viewRules(rec[c(12, 9, 14, 15, 16, 21, 8, 25, 19, 7, 11, 13),])
viewRules(rec[c(22, 10, 17, 20, 18, 23),])

# extract all objects with those objects

df_tmp1 <- filter(dat, c(V85 == "T" & V86 == "I" & V87 == "A" & V88 == "S" & V89 == "V"))

df_tmp2 <- filter(dat, c(V27 == "M" & V208 == "N" & V100 == "I" & V222 == "G" & V63 == "K"
                       & V54 == "L" & V6 == "I" & V85 == "A" & V111 == "E" & V89 == "S"))

df_tmp1[,c("True", "Prediction")] %>% table
df_tmp2[,c("True", "Prediction")] %>% table


# run Rosetta again to find further differences that might help to distinguish between those objects

ros_tmp1 <- rosetta(df_tmp1[,1:(dim(df_tmp1)[2] - 1)], discrete = T, underSample = T, reducer = "Johnson")
ros_tmp1$quality

rec_tmp1 <- recalculateRules(df_tmp1[,1:(dim(df_tmp1)[2] - 1)], ros_tmp1$main, discrete = T)

viewRules(rec_tmp1[rec_tmp1$decision == 1,])
viewRules(rec_tmp1[rec_tmp1$decision == 0,])

# run Rosetta again to find further differences that might help to distinguish between those objects

ros_tmp2 <- rosetta(df_tmp2[,1:(dim(df_tmp2)[2] - 1)], discrete = T, underSample = T, reducer = "Johnson")
ros_tmp2$quality

rec_tmp2 <- recalculateRules(df_tmp2[,1:(dim(df_tmp2)[2] - 1)], ros_tmp2$main, discrete = T)

viewRules(rec_tmp2[rec_tmp2$decision == 1,])
viewRules(rec_tmp2[rec_tmp2$decision == 0,])


