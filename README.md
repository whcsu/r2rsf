# Survival Forests with R-Squared Splitting Rules

Wang Hong, Chen Xiaolin, and Li Gang. Survival Forests with R-Squared Splitting Rules. Journal of Computational Biology. December 2017, https://doi.org/10.1089/cmb.2017.0107

 We propose a survival forest approach with trees constructed using a novel pseudo  R squared  splitting rules based on the following paper:  
 
 Li, G. and Wang, X. 2016. Prediction accuracy measures for a nonlinear model and for right-censored time-to-event data. arXiv preprint arXiv:1611.03063
 
The proposed R-squared random survival forest(R2RSF) is implemented within the framework of the ”ranger” R package (Wright and Ziegler, 2017).

Download the package via github and the usage is similar to that ranger


library(rangernew)

library(survival)

#veteran DATA

data(veteran, package = "survival")

#index of time and censoring status

rii=c(3,4)

mydata=na.omit(veteran)

mydata[,2]=as.numeric(veteran[,2])

colnames(mydata)[rii]=c("time","status")

n=dim(mydata)[1]


r2rsf=rangernew(Surv(time, status) ~ ., data = mydata,splitrule = "r2",num.trees = 500,num.threads = 2)      

