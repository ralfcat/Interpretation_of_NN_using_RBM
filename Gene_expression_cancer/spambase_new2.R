# Clear the environment and set the working directory
rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
set.seed(0)

# Load necessary packages
library(dplyr)
library(R.ROSETTA)

# ======================= Name Preprocessing ==============================

# Load the dataset with original names preserved
data_original <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train2_binned.csv", header = TRUE, check.names = FALSE)

# Load the dataset with R's default name checking
data_modified <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train2_binned.csv", header = TRUE, check.names = TRUE)

# Create a dataframe that maps original names to modified names
name_comparison <- data.frame(
  Original_Names = names(data_original),
  Modified_Names = names(data_modified)
)

# Print the comparison dataframe
print(name_comparison)

# ======================= Build RBM on NN Output ==============================

# Load data
dat <- read.csv("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/Spambase/spambase_train2_binned.csv", header = TRUE, colClasses = "character")
dat <- rename_columns_based_on_comparison(dat, name_comparison)  # Assuming function definition below

# Rename columns for clarity
names(dat)[ncol(dat) - 1] <- "True"
names(dat)[ncol(dat)] <- "Prediction"

# Extract true labels and prepare data frames
truth <- dat$True
df <- dat[, -which(names(dat) %in% c("True", "Prediction"))]
df$Prediction <- dat$Prediction

# Run Rosetta on adjusted data frame
ros <- rosetta(df, discrete = TRUE, underSample = TRUE, reducer = "Johnson")
rev_df <- dat[, rev(names(df))]
rev_ros <- rosetta(rev_df, discrete = TRUE, underSample = TRUE, reducer = "Johnson")

comb <- data.frame(rbind(ros$main, rev_ros$main))
testie <- recalculateRules(df, ros$main, discrete = TRUE)
rec <- recalculateRules(df, comb, discrete = TRUE)
rec <- distinct(rec[rec$pValue <= 0.05, ])
testie_rec <- distinct(testie[testie$pValue <= 0.05, ])

viewRules(testie_rec)
viewRules(rec)

# Function to replace column names based on the comparison dataframe
rename_columns_based_on_comparison <- function(df, name_comparison) {
  names(df) <- sapply(names(df), function(x) {
    idx <- which(name_comparison$Modified_Names == x)
    if (length(idx) == 1) return(name_comparison$Original_Names[idx])
    else return(x)
  })
  return(df)
}

# ======================== Re-run Rosetta to Identify Differences ==============================

# Filter and prepare data for further analysis
df_tmp1 <- filter(dat, char_freq_ == 0 & word_freq_money == 0 & word_freq_your == 0 & word_freq_over == 0 & word_freq_our == 0)
df_tmp1 <- rename_columns_based_on_comparison(df_tmp1, name_comparison)  # Ensure column names are correct

# Run Rosetta on filtered data
ros_tmp1 <- rosetta(df_tmp1[,1:(ncol(df_tmp1) - 1)], discrete = TRUE, underSample = TRUE, reducer = "Johnson")
ros_tmp1$quality
rec_tmp1 <- recalculateRules(df_tmp1[,1:(ncol(df_tmp1) - 1)], ros_tmp1$main, discrete = TRUE)

viewRules(rec_tmp1[rec_tmp1$decision == 1,])
viewRules(rec_tmp1[rec_tmp1$decision == 0,])

