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
dat <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/data/NS1/Pred_labels_train2.csv", header = T, colClasses = "character")

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
testie <- recalculateRules(df,ros$main, discrete = T)
# combine rules and recalculate their quality measures
comb <- data.frame(rbind(ros$main, rev_ros$main, centre_ros$main))
rec <- recalculateRules(df, comb, discrete = T)
rec <- distinct(rec[rec$pValue<=0.05,])

testie_rec <- distinct(testie[testie$pValue<=0.05,])

# check rules
viewRules(rec)

viewRules(testie_rec)

# ======================== Test RBM on NN output ==============================

# load test set
test <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/data/NS1/Pred_labels_test.csv", colClasses = "character", header = T)

# extract data and NN label
test_truth <- test[,(dim(test)[2] - 1)]
test_df <- test[,c(1:(dim(test)[2] - 2), dim(test)[2])]

names(test_df)[dim(test_df)[2]] <- "Prediction"

# predict classes for test data
pred <- predictClass(test_df[,1:(dim(test_df)[2] - 1)], rec, discrete = T, validate = T, defClass = test_df[,dim(test_df)[2]], normalizeMethod = "rulnum")
pred$accuracy
table(pred$out[,c("currentClass", "predictedClass")])

pred_test <- predictClass(test_df[,1:(dim(test_df)[2] - 1)], testie_rec, discrete = T, validate = T, defClass = test_df[,dim(test_df)[2]], normalizeMethod = "rulnum")
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
#viewRules(rec[c(12, 9, 14, 15, 16, 21, 8, 25, 19, 7, 11, 13),])
viewRules(rec[c(22, 10, 17, 20, 18, 23),])

viewRules(rec[c(23,21,25,27,30,31,12,11,18,22),])

# extract all objects with those objects

df_tmp1 <- filter(dat, c(P85 == "T" & P86 == "I" & P87 == "A" & P88 == "S" & P89 == "V"))

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



####### Compare the misclassified object of the NN to the misclassified of the RBM ##########

rbm_predictions <- pred$out$predictedClass

# Misclassified objects by the NN (original)
nn_misclassified <- which(test_df$Prediction != test_truth)

# Misclassified objects by the RBM
rbm_misclassified <- which(rbm_predictions != test_df$Prediction)

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
pushViewport(viewport(x = 1, y = 1.2, just = c("right", "top")))
grid.draw(legend_grob)
popViewport()


# Create the legend grob and draw it
legend_grob <- create_legend_grob()
pushViewport(viewport(x = 0.5, y = 0.5, just = c("right", "top")))
grid.draw(legend_grob)
popViewport()

