#Set the working directory to 1 level below 'data'
#setwd("/Users/jun/Documents/424/a2")
training_data <- read.table("data/intersected_final_chr1_cutoff_20_train_revised.bed", sep="\t")
test_data <- read.table("data/intersected_final_chr1_cutoff_20_test.bed", sep="\t")
sample_data <- read.table("data/intersected_final_chr1_cutoff_20_sample.bed", sep="\t")

