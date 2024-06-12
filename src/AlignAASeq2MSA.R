# Flush r-studio and set working directory 
rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
set.seed(0)

# load the necessary packages
# Install packages
install.packages("Biostrings")
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("msa")

install.packages("dplyr")
install.packages("ape")

# Load packages
library(Biostrings)
library(msa)
library(dplyr)
library(ape)


# read the MSA
df <- read.csv("../data/NS1/NS1.csv", header = T, colClasses = "character")

# extract the objects of serotype H5N1
serotype <- unlist(lapply(df$id, function(x){unlist(strsplit(x, '[*]'))[5]}))
h5n1 <- df[which(serotype=="H5N1"),]
msa <- h5n1[,2:(dim(h5n1)[2]-1)]

# substitute spaces to prevent false parsing when creating StringSet
row.names(msa) <- lapply(h5n1[,1], function(x){gsub(" ", "", x)})

# substitute gap symbol to prevent false parsing when creating StringSet
msa[msa == "?"] = "-"

# save MSA as AAStringSet
alignment_aa <- AAStringSet(apply(msa, 1, function(x){
  gsub('[, ]', '', toString(x))}) %>% unlist)


# read the PDB AA sequence
seq_fasta <- read.FASTA("../data/NS1/rcsb_pdb_3F5T.fasta", type = "AA")
protein_seq <- gsub(", ", '', as.character(seq_fasta) %>% unname %>%
                      unlist %>% toString)

# save sequence in AAStringSet
protein_seq_aa <- AAStringSet(protein_seq)

# compute the MSA based on both AAStringSets
aligned_seq <- msa(c(alignment_aa, protein_seq_aa))

# get position of new aligned sequence
pos <- which(!aligned_seq@unmasked@ranges@NAMES  %in% alignment_aa@ranges@NAMES)

# check that there was no change in the previous alignment
tmp <- unmasked(aligned_seq)[-pos]
!any(tmp[order(names(tmp), decreasing = T),] !=
       alignment_aa[order(names(alignment_aa), decreasing = T),])

# print to see the newly aligned sequence in mid of the previous MSA
print(as(AAMultipleAlignment(unmasked(aligned_seq)[(pos-5):(pos+5)]),
                "MsaAAMultipleAlignment"), show = "complete")

# extract the newly aligned sequence as list
seq <- strsplit(unmasked(aligned_seq)[pos] %>% as.character, '')%>% 
  unlist %>% unname

# save aligned sequence to fasta file
write.csv(seq, "../data/NS1/alignedSeq.fa")

