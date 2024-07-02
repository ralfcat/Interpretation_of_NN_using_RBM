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
dat <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_train2_binned.csv", header = T, colClasses = "character")

# Check column names
print(names(dat))

# Summary to check for any obvious issues in data
summary(dat)


# change names of truth and prediction
names(dat)[dim(dat)[2] - 1] = "True"
names(dat)[dim(dat)[2]] = "Prediction"

# extract true labels
truth <- dat[,(dim(dat)[2] - 1)]

# remove truth label from data frame
df <- dat[,c(1:(dim(dat)[2] - 2), dim(dat)[2])]


# run Rosetta on all three data frame
ros <- rosetta(df, discrete = T, underSample = T, reducer = "Johnson")


testie <- recalculateRules(df,ros$main, discrete = T)
# combine rules and recalculate their quality measures
comb <- data.frame(rbind(ros$main, rev_ros$main, centre_ros$main))
rec <- recalculateRules(df, comb, discrete = T)
rec <- distinct(rec[rec$pValue<=0.05,])

testie_rec <- distinct(testie[testie$pValue<=0.05,])

# check rules
viewRules(rec)

viewRules(testie_rec)

viewRules(testie_rec[testie_rec$decision == 1,])
viewRules(testie_rec[testie_rec$decision == 0,])



# ======================== Test RBM on NN output ==============================

# load test set
test <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Gene_expression_cancer/data/cancer_genexp_test_binned.csv", colClasses = "character", header = T)

# extract data and NN label
test_truth <- test[,(dim(test)[2] - 1)]
test_df <- test[,c(1:(dim(test)[2] - 2), dim(test)[2])]

names(test_df)[dim(test_df)[2]] <- "Prediction"



# predict classes for test data
pred <- predictClass(test_df[,1:(dim(test_df)[2] - 1)], testie_rec, discrete = T, validate = T, defClass = test_df[,dim(test_df)[2]], normalizeMethod = "rulnum")
pred$accuracy
table(pred$out[,c("currentClass", "predictedClass")])

pred_test <- predictClass(test_df[,1:(dim(test_df)[2] - 1)], testie_rec, discrete = T, validate = T, defClass = test_df[,dim(test_df)[2]], normalizeMethod = "rulnum")
pred_test$accuracy
table(pred_test$out[,c("currentClass", "predictedClass")])
# ================= Analysis of wrongly classified objects ====================

# extract wrongly classified objects
wrongObj <- which(df[,dim(df)[2]] != truth)

# get supportSet as list
supportSet <- lapply(testie_rec$supportSetRHS, function(x){as.numeric(unlist(strsplit(x, ",")))})

# create heatmap to identify misclassified objects in support set of rules
rules <- list()
for(i in wrongObj){
  tmp = which(unlist(lapply(supportSet, function(x){(i %in% x)})))
  rules[[length(rules) + 1]] <- as.numeric(row.names(testie_rec[tmp,]))
}

r <- unlist(rules) %>% unique

x = lapply(rules, function(x){as.numeric(r %in% x)}) %>% data.frame
names(x) = wrongObj
row.names(x) <- r

heatmap(as.matrix(x), scale = "none", Colv = F, col = c("white", "black"),
        xlab = "Object Number", ylab = "Rule Rank")

# based on heatmap inspect rules 
hmap_rules <-viewRules(testie_rec[c(7,127),])