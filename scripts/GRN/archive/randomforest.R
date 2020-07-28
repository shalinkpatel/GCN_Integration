library(GENIE3)
library(doParallel)
library(doRNG)

exprMat <- read.csv("~/scratch/HEPG2_expression_matrix_imputed.tsv", sep = '\t')

weightMat <- GENIE3(as.matrix(exprMat), nCores=32)
write.table(weightMat, "~/data/spate116/GCN/GRN.tsv", sep='\t')

