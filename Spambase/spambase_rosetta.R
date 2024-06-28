# Flush r-studio and set working directory 
rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
set.seed(0)

# load necessary packages
library(dplyr)
library(R.ROSETTA)

# ======================= name preprocessing ==============================

# Load the dataset with original names preserved
data_original <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train2_binned.csv", header = T,check.names = FALSE)

# Load the dataset with R's default name checking
data_modified <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train2_binned.csv", header = T, check.names = TRUE)

# Create a dataframe that maps original names to modified names
name_comparison <- data.frame(
  Original_Names = names(data_original),
  Modified_Names = names(data_modified)
)

# Print the comparison dataframe
print(name_comparison)



# ======================= Build RBM on NN output ==============================

# load data
dat <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train2_binned.csv", header = T, colClasses = "character")

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

######### Replacing the wrong names with the correct ones ###############
# Split the concatenated names into a list of vectors
features_list <- strsplit(testie_rec$features, ",")

# Function to find original name for each modified name
get_original_name <- function(modified_name) {
  # Trim whitespace which might be left from splitting
  modified_name <- trimws(modified_name)
  # Find the match in name_comparison and return the original name
  index <- match(modified_name, name_comparison$Modified_Names)
  if (!is.na(index)) {
    return(name_comparison$Original_Names[index])
  } else {
    return(modified_name)  # Return the modified name if no original found
  }
}

# Apply the function to each element in the list and recombine names
corrected_features <- sapply(features_list, function(feature_vector) {
  original_names <- sapply(feature_vector, get_original_name)
  paste(original_names, collapse = ",")
})

# Replace the original features with corrected ones
testie_rec$features <- corrected_features

# Check the results
print(testie_rec$features)



# ======================== Test RBM on NN output ==============================

# load test set
test <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_test_binned.csv", colClasses = "character", header = T, check.names = F)

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
