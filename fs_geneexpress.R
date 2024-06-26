install.packages("rmcfs")
install.packages("data.table")

library(rmcfs)
library(data.table)

# Loading data
setwd("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM")
data <- fread("/Users/victorenglof/Documents/GitHub/Interpretation_of_NN_using_RBM/merged_dataset.csv")
data <- as.data.frame(data)

# Feature selection
res <- mcfs(Class~., data, cutoffPermutations = 20, projections = 25000,projectionSize = 143, threadsNumber = 20)
saveRDS(res, )

# Fetching rankings
mcfs_res <- res$RI[1:res$cutoff_value,]

# Selecting the first 160 attributes
attr <- mcfs_res[1:499, "attribute"]

# Reducing the dataset
r_data <- data[, c(attr, "Class")]

# Creating csv
write.csv(r_data, file = "TCGA_GTEx_reduced.csv", row.names = FALSE)
