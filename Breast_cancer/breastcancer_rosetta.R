# Flush r-studio and set working directory 
rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
set.seed(0)

# Load necessary packages
library(dplyr)
library(R.ROSETTA)

# ======================= Build RBM on NN output ==============================

# Load data
dat <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Breast_cancer/breastcancer_train2_binned.csv", header = TRUE, colClasses = "character")

# Check column names
print(names(dat))

# Summary to check for any obvious issues in data
summary(dat)

# Change names of truth and prediction
names(dat)[dim(dat)[2] - 1] <- "True"
names(dat)[dim(dat)[2]] <- "Prediction"

# Extract true labels
truth <- dat[, (dim(dat)[2] - 1)]

# Remove truth label from data frame
df <- dat[, c(1:(dim(dat)[2] - 2), dim(dat)[2])]

# Run Rosetta on the data frame
ros <- rosetta(df, discrete = TRUE, underSample = TRUE, reducer = "Johnson")

testie <- recalculateRules(df, ros$main, discrete = TRUE)

# Combine rules and recalculate their quality measures
comb <- data.frame(rbind(ros$main, ros$main, ros$main)) # Adjust as necessary
rec <- recalculateRules(df, comb, discrete = TRUE)
rec <- distinct(rec[rec$pValue <= 0.05, ])

testie_rec <- distinct(testie[testie$pValue <= 0.05, ])

# Check rules
viewRules(rec)
viewRules(testie_rec)
viewRules(testie_rec[testie_rec$decision == 1, ])
viewRules(testie_rec[testie_rec$decision == 0, ])

# ======================== Test RBM on NN output ==============================

# Load test set
test <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Breast_cancer/breastcancer_test_binned.csv", colClasses = "character", header = TRUE)

# Extract data and NN label
test_truth <- test[, (dim(test)[2] - 1)]
test_df <- test[, c(1:(dim(test)[2] - 2), dim(test)[2])]

names(test_df)[dim(test_df)[2]] <- "Prediction"

# Predict classes for test data
pred <- predictClass(test_df[, 1:(dim(test_df)[2] - 1)], testie_rec, discrete = TRUE, validate = TRUE, defClass = test_df[, dim(test_df)[2]], normalizeMethod = "rulnum")
cat("RBM Prediction Accuracy:", pred$accuracy, "\n")
print(table(pred$out[, c("currentClass", "predictedClass")]))

pred_test <- predictClass(test_df[, 1:(dim(test_df)[2] - 1)], testie_rec, discrete = TRUE, validate = TRUE, defClass = test_df[, dim(test_df)[2]], normalizeMethod = "rulnum")
cat("RBM Test Prediction Accuracy:", pred_test$accuracy, "\n")
print(table(pred_test$out[, c("currentClass", "predictedClass")]))

# ================= Analysis of wrongly classified objects ====================

# Extract wrongly classified objects
wrongObj <- which(test_df$Prediction != test_truth)
cat("Number of wrongly classified objects:", length(wrongObj), "\n")

if (length(wrongObj) < 2) {
  stop("Not enough misclassified objects to create a heatmap.")
}

# Get supportSet as list
supportSet <- lapply(testie_rec$supportSetRHS, function(x) { as.numeric(unlist(strsplit(x, ","))) })

# Create heatmap to identify misclassified objects in support set of rules
rules <- list()
for (i in wrongObj) {
  tmp <- which(unlist(lapply(supportSet, function(x) { (i %in% x) })))
  if (length(tmp) > 0) {
    rules[[length(rules) + 1]] <- as.numeric(row.names(testie_rec[tmp, ]))
  }
}

r <- unlist(rules) %>% unique

if (length(r) < 2) {
  stop("Not enough rules to create a heatmap.")
}

x <- lapply(rules, function(x) { as.numeric(r %in% x) }) %>% data.frame
names(x) <- wrongObj
row.names(x) <- r

# Ensure there are enough rows and columns to create a heatmap
if (nrow(x) < 2 | ncol(x) < 2) {
  stop("The matrix 'x' must have at least 2 rows and 2 columns to create a heatmap.")
}

heatmap(as.matrix(x), scale = "none", Colv = FALSE, col = c("white", "black"),
        xlab = "Object Number", ylab = "Rule Rank")

# Based on heatmap inspect rules
hmap_rules <- viewRules(testie_rec[c(7, 127), ])



##### view rules ####


viewRules(testie_rec[r, ])


##### RE RUN ROSETTA BASED ON THE RULES #####

df_tmp1 <- filter(dat, c(concave.points_worst == "Low"))


df_tmp1[,c("True", "Prediction")] %>% table
df_tmp2[,c("True", "Prediction")] %>% table


# run Rosetta again to find further differences that might help to distinguish between those objects

ros_tmp1 <- rosetta(df_tmp1[,1:(dim(df_tmp1)[2] - 1)], discrete = T, underSample = T, reducer = "Johnson")
ros_tmp1$quality

rec_tmp1 <- recalculateRules(df_tmp1[,1:(dim(df_tmp1)[2] - 1)], ros_tmp1$main, discrete = T)

viewRules(rec_tmp1[rec_tmp1$decision == 1,])
viewRules(rec_tmp1[rec_tmp1$decision == 0,])


