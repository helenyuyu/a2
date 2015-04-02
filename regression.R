#Set the working directory to 1 level below 'data'
setwd("/Users/jun/Documents/424/a2")
#training_data <- read.table("data/intersected_final_chr1_cutoff_20_train_revised.bed", sep="\t")
#test_data <- read.table("data/intersected_final_chr1_cutoff_20_test.bed", sep="\t")
#sample_data <- read.table("data/intersected_final_chr1_cutoff_20_sample.bed", sep="\t")

library(betareg)

fill_in_row <- function(row) {
  count <- 0;
  sum <- 0;
  for (i in row) {
    if (!is.nan(i)) {
      count <- count + 1;
      sum <- sum + 1;
    }
  }
  avg <- sum / count;
  for (i in 0:(length(row))) {
    if (is.nan(row[i])) {
      row[i] = avg;
    }
  }
  return (row);
}

training_sub <- training_data[,5:37]



gy <- betareg()
summary(gy)




