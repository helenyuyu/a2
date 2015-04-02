#Set the working directory to 1 level below 'data'
setwd("/Users/jun/Documents/424/a2")
#training_data <- read.table("data/intersected_final_chr1_cutoff_20_train_revised.bed", sep="\t")
#test_data <- read.table("data/intersected_final_chr1_cutoff_20_test.bed", sep="\t")
#sample_data <- read.table("data/intersected_final_chr1_cutoff_20_sample.bed", sep="\t")
#> known <- read.table("known")

library(betareg)

#known <- read.table("known")
#unknown_target <- read.table("unknown_target")
#unknown_vector <- read.table("unknown_vector")


gy <- betareg(V34 ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+
    V11+V12+V13+V14+V15+V16+V17+V18+V19+V20+V21+V22+V23+V24+V25+
    V26+V27+V28+V29+V30+V31+V32+V33, data = known)
#summary(gy)
predicted_target <- predict(gy, type="response", newdata=unknown_vector)
RMSE <- sqrt(mean((predicted_target-unknown_target)^2))

meanPred <- mean(predicted_target)
SSRes <- sum((predicted_target - unknown_target)^2)
SSTot <- sum((predicted_target - meanPred)^2)
RSquared <- 1 - (SSRes)/(SSTot)

