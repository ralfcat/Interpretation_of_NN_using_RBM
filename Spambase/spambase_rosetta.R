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
rev_df <- dat[,c((dim(dat)[2] - 2):1, dim(dat)[2])]

# run Rosetta on all three data frame
ros <- rosetta(df, discrete = T, underSample = T, reducer = "Johnson")
rev_ros <- rosetta(rev_df, discrete = T, underSample = T, reducer = "Johnson")

comb <- data.frame(rbind(ros$main, rev_ros$main))


testie <- recalculateRules(df,ros$main, discrete = T)

rec <- recalculateRules(df, comb, discrete = T)
rec <- distinct(rec[rec$pValue<=0.05,])

testie_rec <- distinct(testie[testie$pValue<=0.05,])

viewRules(testie_rec)
viewRules(rec)
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


######### Replacing the wrong names with the correct ones ###############
# Split the concatenated names into a list of vectors
features_list <- strsplit(rec$features, ",")

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
rec$features <- corrected_features

# Check the results
print(rec$features)


# ======================== Test RBM on NN output ==============================

# load test set
test <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_test_binned.csv", colClasses = "character", header = T, check.names = F)

# extract data and NN label
test_truth <- test[,(dim(test)[2] - 1)]
test_df <- test[,c(1:(dim(test)[2] - 2), dim(test)[2])]

names(test_df)[dim(test_df)[2]] <- "Prediction"



# predict classes for test data
pred <- predictClass(test_df[,1:(dim(test_df)[2] - 1)], rec, discrete = T, validate = T, defClass = test_df[,dim(test_df)[2]], normalizeMethod = "rulnum")
pred$accuracy
table(pred$out[,c("currentClass", "predictedClass")])

pred_test <- predictClass(test_df[,1:(dim(test_df)[2] - 1)], rec, discrete = T, validate = T, defClass = test_df[,dim(test_df)[2]], normalizeMethod = "rulnum")
pred_test$accuracy
table(pred_test$out[,c("currentClass", "predictedClass")])
# ================= Analysis of wrongly classified objects ====================

# extract wrongly classified objects
wrongObj <- which(df[,dim(df)[2]] != truth)


# Identify false positives and false negatives
falsePos <- which(df[,dim(df)[2]] == 1 & truth == 0)
falseNeg <- which(df[,dim(df)[2]] == 0 & truth == 1)

# Create color vector for ColSideColors
colSideColors <- rep("white", nrow(df))
colSideColors[falsePos] <- "green"
colSideColors[falseNeg] <- "red"

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

# Plot heatmap with ColSideColors
heatmap(as.matrix(x), scale = "none", Colv = F, col = c("white", "black"),
        ColSideColors = colSideColors[wrongObj],
        xlab = "Object Number", ylab = "Rule Rank")

# based on heatmap inspect rules 
hmap_rules <-viewRules(rec[c(72,70,98,115,2),])

print(hmap_rules)

########### MISCLASSIFICATION ANALYSIS BASED ON BOTTOM LEFT CLUSTER ###############


df_tmp1 <- filter(dat, `char_freq_!` == 0 & word_freq_money == 0 & word_freq_your == 0 & word_freq_over == 0 & word_freq_our == 0)




df_tmp1[,c("True", "Prediction")] %>% table

# run Rosetta again to find further differences that might help to distinguish between those objects

ros_tmp1 <- rosetta(df_tmp1[,1:(dim(df_tmp1)[2] - 1)], discrete = T, underSample = T, reducer = "Johnson")
ros_tmp1$quality

rec_tmp1 <- recalculateRules(df_tmp1[,1:(dim(df_tmp1)[2] - 1)], ros_tmp1$main, discrete = T)

viewRules(rec_tmp1[rec_tmp1$decision == 1,])
viewRules(rec_tmp1[rec_tmp1$decision == 0,])

####### Compare the misclassified object of the NN to the misclassified of the RBM ##########

rbm_predictions <- pred$out$predictedClass

# Misclassified objects by the NN (original)
nn_misclassified <- which(test_df$Prediction != test_truth)

# Misclassified objects by the RBM
rbm_misclassified <- which(rbm_predictions != test_truth)

# Find common misclassified objects
common_misclassified <- intersect(nn_misclassified, rbm_misclassified)

# Find misclassified objects unique to each model
nn_unique_misclassified <- setdiff(nn_misclassified, rbm_misclassified)
rbm_unique_misclassified <- setdiff(rbm_misclassified, nn_misclassified)

# Summary of comparison
cat("Number of misclassified objects by NN:", length(nn_misclassified), "\n")
cat("Number of misclassified objects by RBM:", length(rbm_misclassified), "\n")
cat("Number of common misclassified objects:", length(common_misclassified), "\n")
cat("Number of unique misclassified objects by NN:", length(nn_unique_misclassified), "\n")
cat("Number of unique misclassified objects by RBM:", length(rbm_unique_misclassified), "\n")

# Print the exact misclassified objects
cat("Misclassified objects by NN:\n", nn_misclassified, "\n")
cat("Misclassified objects by RBM:\n", rbm_misclassified, "\n")
cat("Common misclassified objects:\n", common_misclassified, "\n")
cat("Unique misclassified objects by NN:\n", nn_unique_misclassified, "\n")
cat("Unique misclassified objects by RBM:\n", rbm_unique_misclassified, "\n")

# Check if "Number of misclassified objects by NN" and "Number of common misclassified objects" are the same instances
if (all(sort(nn_misclassified) == sort(common_misclassified))) {
  cat("The misclassified objects by NN are the same as the common misclassified objects.\n")
} else {
  cat("The misclassified objects by NN are NOT the same as the common misclassified objects.\n")
}

# Extract the misclassified instances
misclassified_instances <- test[common_misclassified, ]

# Check for NAs and remove them if present
misclassified_instances <- misclassified_instances[complete.cases(misclassified_instances), ]
names(misclassified_instances)[dim(misclassified_instances)[2] - 1] = "True"
names(misclassified_instances)[dim(misclassified_instances)[2]] = "Prediction"

# extract true labels
truth <- misclassified_instances[,(dim(misclassified_instances)[2] - 1)]

# remove truth label from data frame
misclassified_instances <- misclassified_instances[,c(1:(dim(misclassified_instances)[2] - 2), dim(misclassified_instances)[2])]

# Convert columns to appropriate types if necessary
misclassified_instances[] <- lapply(misclassified_instances, function(x) {
  if (is.character(x)) {
    return(as.factor(x))
  } else {
    return(x)
  }
})

# Retrain the RBM on the misclassified instances
ros_misclassified <- rosetta(misclassified_instances, discrete = TRUE, underSample = TRUE, reducer = "Genetic")

# Extract and view the rules used for misclassified instances
rec_misclassified <- recalculateRules(misclassified_instances[, 1:(dim(misclassified_instances)[2] - 1)], ros_misclassified$main, discrete = TRUE)
cat("Rules used by RBM for misclassified objects:\n")
viewRules(rec_misclassified)

##### Running Rosetta on the misclassified objects #####


# Extract the misclassified instances
misclassified_instances <- test[common_misclassified, ]

# Check for NAs and remove them if present
misclassified_instances <- misclassified_instances[complete.cases(misclassified_instances), ]
names(misclassified_instances)[dim(misclassified_instances)[2] - 1] = "True"
names(misclassified_instances)[dim(misclassified_instances)[2]] = "Prediction"

# extract true labels
truth <- misclassified_instances[,(dim(misclassified_instances)[2] - 1)]

# remove truth label from data frame
misclassified_instances <- misclassified_instances[,c(1:(dim(misclassified_instances)[2] - 2), dim(misclassified_instances)[2])]

# Convert columns to appropriate types if necessary
misclassified_instances[] <- lapply(misclassified_instances, function(x) {
  if (is.character(x)) {
    return(as.factor(x))
  } else {
    return(x)
  }
})

# Retrain the RBM on the misclassified instances
ros_misclassified <- rosetta(misclassified_instances, discrete = TRUE, underSample = TRUE, reducer = "Johnson")

# Extract and view the rules used for misclassified instances
rec_misclassified <- recalculateRules(misclassified_instances[, 1:(dim(misclassified_instances)[2] - 1)], ros_misclassified$main, discrete = TRUE)
cat("Rules used by RBM for misclassified objects:\n")
viewRules(rec_misclassified)



#### Plotting the misclassified objects #####
# Install and load necessary packages
if (!require("VennDiagram")) {
  install.packages("VennDiagram")
}
library(VennDiagram)
library(grid)

# Create a Venn diagram and save to a temporary file
venn.plot <- venn.diagram(
  x = list(
    "NN Misclassified" = nn_misclassified,
    "RBM Misclassified" = rbm_misclassified
  ),
  category.names = c("NN Misclassified", "RBM Misclassified"),
  fill = c("red", "blue"),
  alpha = 0.5,
  cat.cex = 0, # Set to 0 to hide the names of the circles
  cex = 1.5,
  scaled = TRUE, # Scale the circles proportionally to the number of elements
  main = "Venn Diagram of Misclassified Objects by NN and RBM",
  filename = NULL # Use NULL to prevent automatic saving to a file
)

# Function to create a legend grob
create_legend_grob <- function() {
  legend_grob <- legendGrob(
    labels = c("NN Misclassified", "RBM Misclassified"),
    pch = 15,
    gp = gpar(col = c("red", "blue"), fill = c("red", "blue")),
    nrow = 1,
    byrow = TRUE
  )
  legend_grob
}

# Plot the Venn diagram in RStudio and add legend
grid.newpage()
grid.draw(venn.plot)

# Create the legend grob and draw it
legend_grob <- create_legendgrob()
pushViewport(viewport(x = 1, y = 1.4, just = c("right", "top")))
grid.draw(legend_grob)
popViewport()


