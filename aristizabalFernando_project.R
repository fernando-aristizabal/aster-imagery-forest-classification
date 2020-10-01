##########################################################################################################
#################################### ABE 6933 FINAL CLASS PROJECT ########################################
##########################################################################################################
#
#   Name: Aristizabal, Fernando
#   Semester: Fall 2016
#   Dataset: https://archive.ics.uci.edu/ml/datasets/Forest+type+mapping
#   Paper: 'Johnson, B., Tateishi, R., Xie, Z., 2012. Using geographically-weighted variables for ...
#           image classification. Remote Sensing Letters, 3 (6), 491-499'
#
#   Note: All lines below have been tested and verified on Linux Ubuntu  16.04 LTS. Parallelization may 
#         have difficulties if utilizing on another platform or less than two cores.
#
#
##########################################################################################################
####################################### ATTN: REQUIRES USER INPUT ########################################
##########################################################################################################
#
# set the working directory to the same as this script
setwd("/home/fernandoa/Documents/ed/16f/sl/proj/aristizabal_fernando")
#
# Note: Ensure working directory contains all the sub-directories from the ...
#       zip file including "./functions" and "./data".
#
##########################################################################################################
############################################# Prerequisites ##############################################
##########################################################################################################

# clears environment
rm(list = ls())

# set seed
set.seed(1)

# source all functions
sapply(paste0(getwd(),'/functions/',list.files('./functions/')),source,.GlobalEnv)

# install and load all necessary packages
install_load(c("tree","glmnet","randomForest","MASS","class","leaps",
               "e1071","gbm","parallel"))

# import training and testing data
tr <- read.csv("./data/training.csv",header=T)
tst <- read.csv("./data/testing.csv",header=T)

# create list to store final answers
parameter <- list() # list of all parameter sweeps results for all models
accuracy <- list() # testing accuracy list for every model
confusion <- list() # testing confusion matrix for every model

# creates parallel cluster
clus <- suppressWarnings(makeCluster(detectCores()))


##########################################################################################################
########################################## Classification Models #########################################
##########################################################################################################

##########################################################################################################
## Linear Discriminant Analysis

model <- lda(as.numeric(class) ~ . , data=tr,method="mve") # train
pred <- predict(model,tst) # predict on test data

accuracy$lda <- mean(pred$class == as.factor(as.numeric(tst$class))) # document test accuracy
confusion$lda <- table(pred=factor(pred$class,labels=c("d","h","o","s")),tst=tst$class) # confusion matrix

##########################################################################################################
## Quadratic Discriminant Analysis

model <- qda(as.numeric(class) ~ . , data=tr) # train
pred <- predict(model,tst) # predict on test

accuracy$qda <- mean(pred$class == as.factor(as.numeric(tst$class))) # test accuracy
confusion$qda <- table(pred=factor(pred$class,labels=c("d","h","o","s")),tst=tst$class) # confusion matrix

##########################################################################################################
## k-Nearest Neighbor

parameter$knn <- data.frame(k=1:20,accuracy=rep(NaN,20)) # initialize k

# generate function to CV KNN
cvKNN <- function(k) {
  model <- knn.cv(tr[,-1],tr$class,k=k)
  return(mean(model == tr$class))
}

parameter$knn$accuracy <- sapply(parameter$knn$k,cvKNN) # CV for every k value

model <- knn(tr[,-1],tst[,-1],tr$class,k=parameter$knn$k[which.max(parameter$knn$accuracy)]) # train with best k
accuracy$knn <- data.frame(k=parameter$knn$k[which.max(parameter$knn$accuracy)], # accuracy on test
                           accuracy=mean(model == tst$class))
confusion$knn <- table(pred=model,tst=tst$class) # confusion matrix

##########################################################################################################
## Linear SVM

parameter$svm$linear <- list(kernel="linear",cost=10^seq(-1,0,0.01),tolerance=10^seq(-10,0,1),epsilon=0.1)

model <- tune(svm, class ~ ., data=tr, ranges=parameter$svm$linear) # tune svm with cv
pred <- predict(model$best.model,newdata=tst) # predict on test set with optimal parameters
accuracy$svm$linear <- data.frame(model$best.parameters[-1],accuracy=mean(pred == tst$class)) # record test accuracy
parameter$svm$linear <- list(cost=parameter$svm$linear$cost, # record cv results
                             costACC=1-model$performances$error[which(model$performances$tolerance==1e-5 & model$performances$epsilon==0.1)],
                             tolerance=parameter$svm$linear$tolerance,
                             toleranceACC=1-model$performances$error[which(model$performances$cost==1 & model$performances$epsilon==0.1)],
                             epsilon=parameter$svm$linear$epsilon,
                             epsilonACC=1-model$performances$error[which(model$performances$cost==1 & model$performances$tolerance==1e-5)])
confusion$svm$linear <- table(pred,tst=tst$class) # confusion matrix

##########################################################################################################
## Radial Basis SVM

parameter$svm$radial <- list(kernel="radial",cost=10^seq(-2,2,0.25),tolerance=1e-5,
                             epsilon=0.1,gamma=10^seq(-2,0,0.05))

model <- tune(svm, class ~ ., data=tr, ranges=parameter$svm$radial) # tune svm with cv
pred <- predict(model$best.model,newdata=tst) # predict on test set with optimal parameters
accuracy$svm$radial <- data.frame(model$best.parameters[-1],accuracy=mean(pred == tst$class)) # record test accuracy
parameter$svm$radial <- list(cost=parameter$svm$radial$cost, # record cv results with other parameters set at default values
                             costACC=1-model$performances$error[which(model$performances$tolerance==1e-5 & model$performances$epsilon==0.1 & model$performances$gamma==10^-1.5)],
                             tolerance=parameter$svm$radial$tolerance,
                             toleranceACC=1-model$performances$error[which(model$performances$cost==1 & model$performances$epsilon==0.1 & model$performances$gamma==10^-1.5)],
                             epsilon=parameter$svm$radial$epsilon,
                             epsilonACC=1-model$performances$error[which(model$performances$cost==1 & model$performances$tolerance==1e-5 & model$performances$gamma==10^-1.5)],
                             gamma=parameter$svm$radial$gamma,
                             gammaACC=1-model$performances$error[which(model$performances$cost==1 & model$performances$tolerance==1e-5 & model$performances$epsilon==0.1)])
confusion$svm$radial <- table(pred,tst=tst$class) # confusion matrix


##########################################################################################################
## Trees

model <- tree(class ~ ., data=tr) # fit tree
cv <- cv.tree(model,K=10,FUN=prune.tree) # cross-validate for best tree size
model <- prune.tree(model,best=cv$size[which.min(cv$dev)]) # prune tree to best size
pred <- predict(model,newdata=tst,type="class") # predict on test set

# record results
parameter$trees <- data.frame(size=cv$size,deviance=cv$dev) 
confusion$trees <- table(pred,tst=tst$class) # confusion matrix
accuracy$trees <- data.frame(size=cv$size[which.min(cv$dev)],accuracy=mean(pred == tst$class))

rm(cv) # clear extraneous variables

##########################################################################################################
## Boosting

# generate array of all possible parameter combinations for grid search
grid <- expand.grid(shrinkage=10^seq(-0.8,-.4,0.05),nTrees=seq(100,800,200),depth=1:2)
#shrinkage=10^seq(-1,-.4,0.05)0.398,nTrees=seq(100,3000,200)3000,depth=1:3(2)

# boost function to be applied to every parameter combinations
boost <- function(parameterGrid) { 
  model <- gbm(class ~ .,data=tr,cv.folds=10,
               shrinkage=parameterGrid[1],n.trees=parameterGrid[2],
               interaction.depth=parameterGrid[3],distribution = "multinomial")
  return(1-(min(model$train.error)/100)) # return cv accuracy for every parameter combination
}

clusterExport(cl=clus,list("gbm","tr")) # export functions and variables to cluster
grid$accuracy <- parApply(cl=clus,grid,1,boost) # hit boosting with all grid rows (parallel computing)
stopCluster(clus) # stops cluster

# vector of accuracies for each parameter while the other parameters are held at fixed values
parameter$boosting <- list(shrinkage=unique(grid$shrinkage),
                           shrinkageACC=grid$accuracy[which(grid$nTrees == 100 & grid$depth == 1)],
                           nTrees=unique(grid$nTrees),
                           nTreesACC=grid$accuracy[which(grid$shrinkage == 0.001 & grid$depth == 1)],
                           depth=unique(grid$depth),
                           depthACC=grid$accuracy[which(grid$shrinkage == 0.001 & grid$nTrees == 100)])

# generate boosting model with best parameter combinations
model <- gbm(class ~ .,data=tr,shrinkage=grid$shrinkage[which.max(grid$accuracy)],
             n.trees=grid$nTrees[which.max(grid$accuracy)],
             interaction.depth=grid$depth[which.max(grid$accuracy)],distribution = "multinomial")

# generate test set predictions
pred <- predict(model,tst,n.trees=grid$nTrees[which.max(grid$accuracy)],type="response") 

# report best parameter combination and corresponding test accuracy
accuracy$boosting <- data.frame(shrinkage=grid$shrinkage[which.max(grid$accuracy)],
                                n.trees=grid$nTrees[which.max(grid$accuracy)],
                                interaction.depth=grid$depth[which.max(grid$accuracy)],
                                accuracy=mean(tst$class == factor(apply(pred,MARGIN=1,which.max),
                                                                  labels=c("d ","h ","o ","s "))))

# confusion matrix
confusion$boosting <- table(pred=factor(apply(pred,MARGIN=1,which.max),labels=c("d ","h ","o ","s ")),
                            tst=tst$class)

rm(grid) # clear extraneous variables

##########################################################################################################
## Random Forests

# tune random forest
model <- tune.randomForest(x=tr[,-1],y=tr$class,mtry=seq(floor(sqrt(ncol(tr))),ncol(tr),1),
                           nodesize=1:5,ntree=seq(50,250,50))
#mtry=seq(floor(sqrt(ncol(tr))),ncol(tr),1)(11),nodesize=1:10(1),ntree=seq(50,500,50)(50)


# save parameters and corresponding accuracy
parameter$randomForest <- list(mtry=unique(model$performance$mtry),
                           mtryACC=1-model$performances$error[which(model$performances$nodesize == 1 & model$performances$ntree == 500)],
                           nodesize=unique(model$performance$nodesize),
                           nodesizeACC=1-model$performances$error[which(model$performances$mtry == 5 & model$performances$ntree == 500)],
                           ntree=unique(model$performance$ntree),
                           ntreeACC=1-model$performances$error[which(model$performances$mtry == 5 & model$performances$nodesize == 1)])

pred <- predict(model$best.model,tst) # predict on test

# report best parameter combination and corresponding test accuracy
accuracy$randomForest <- data.frame(mtry=model$best.parameters$mtry,
                                nodesize=model$best.parameters$nodesize,
                                ntree=model$best.parameters$ntree,
                                accuracy=mean(tst$class == pred))

# confusion matrix
confusion$randomForest <- table(pred,tst=tst$class)

rm(model,pred,clus) # remove extraneous variables

##########################################################################################################
######################################### Plotting CV Results ############################################
##########################################################################################################


## knn #############################

# k
jpeg("./figures/knn.jpeg")
plot(parameter$knn$k,parameter$knn$accuracy,main="10-Fold Cross Validation for KNN",
     xlab="k parameter",ylab="Cross Validation Accuracy")
points(parameter$knn$k[which.max(parameter$knn$accuracy)],max(parameter$knn$accuracy),
       pch=4,col="red",lwd=3)
dev.off()

## trees #############################

# size
jpeg("./figures/trees.jpeg")
plot(parameter$trees$size,parameter$trees$deviance,main="10-Fold CV for Decision Trees",
     xlab="Tree Size Parameter",ylab="Cross Validation Deviance",type="l")
points(parameter$trees$size[which.min(parameter$trees$deviance)],min(parameter$trees$deviance),
       pch=4,col="red",lwd=3)
dev.off()

## linear svm #############################

# cost
jpeg("./figures/svm_linear_cost.jpeg")
plot(parameter$svm$linear$cost,parameter$svm$linear$costACC,main="10-Fold CV for Linear SVM",
     xlab="Cost Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$svm$linear$cost[which.max(parameter$svm$linear$costACC)],max(parameter$svm$linear$costACC),
       pch=4,col="red",lwd=3)
dev.off()

# tolerance
jpeg("./figures/svm_linear_tolerance.jpeg")
plot(parameter$svm$linear$tolerance,parameter$svm$linear$toleranceACC,main="10-Fold CV for Linear SVM",
     xlab="Tolerance Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$svm$linear$tolerance[which.max(parameter$svm$linear$toleranceACC)],max(parameter$svm$linear$toleranceACC),
       pch=4,col="red",lwd=3)
dev.off()


# epsilon
jpeg("./figures/svm_linear_epsilon.jpeg")
plot(parameter$svm$linear$epsilon,parameter$svm$linear$epsilonACC,main="10-Fold CV for Linear SVM",
     xlab="epsilon Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$svm$linear$epsilon[which.max(parameter$svm$linear$epsilonACC)],max(parameter$svm$linear$epsilonACC),
       pch=4,col="red",lwd=3)
dev.off()


## radial svm #############################

# cost
jpeg("./figures/svm_radial_cost.jpeg")
plot(parameter$svm$radial$cost,parameter$svm$radial$costACC,main="10-Fold CV for Radial SVM",
     xlab="Cost Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$svm$radial$cost[which.max(parameter$svm$radial$costACC)],max(parameter$svm$radial$costACC),
       pch=4,col="red",lwd=3)
dev.off()

# tolerance
jpeg("./figures/svm_radial_tolerance.jpeg")
plot(parameter$svm$radial$tolerance,parameter$svm$radial$toleranceACC,main="10-Fold CV for Radial SVM",
     xlab="Tolerance Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$svm$radial$tolerance[which.max(parameter$svm$radial$toleranceACC)],max(parameter$svm$radial$toleranceACC),
       pch=4,col="red",lwd=3)
dev.off()

# epsilon
jpeg("./figures/svm_radial_epsilon.jpeg")
plot(parameter$svm$radial$epsilon,parameter$svm$radial$epsilonACC,main="10-Fold CV for Radial SVM",
     xlab="epsilon Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$svm$radial$epsilon[which.max(parameter$svm$radial$epsilonACC)],max(parameter$svm$radial$epsilonACC),
       pch=4,col="red",lwd=3)
dev.off()

# gamma
jpeg("./figures/svm_radial_gamma.jpeg")
plot(parameter$svm$radial$gamma,parameter$svm$radial$gammaACC,main="10-Fold CV for Radial SVM",
     xlab="Gamma Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$svm$radial$gamma[which.max(parameter$svm$radial$gammaACC)],max(parameter$svm$radial$gammaACC),
       pch=4,col="red",lwd=3)
dev.off()


## boosting #############################

# shrinkage
jpeg("./figures/boosting_shrinkage.jpeg")
plot(parameter$boosting$shrinkage,parameter$boosting$shrinkageACC,main="10-Fold CV for Boosting",
     xlab="Shrinkage Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$boosting$shrinkage[which.max(parameter$boosting$shrinkageACC)],max(parameter$boosting$shrinkageACC,na.rm=T),
       pch=4,col="red",lwd=3)
dev.off()

# number of trees
jpeg("./figures/boosting_nTrees.jpeg")
plot(parameter$boosting$nTrees,parameter$boosting$nTreesACC,main="10-Fold CV for Boosting",
     xlab="Number of Trees Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$boosting$nTrees[which.max(parameter$boosting$nTreesACC)],max(parameter$boosting$nTreesACC),
       pch=4,col="red",lwd=3)
dev.off()

# depth
jpeg("./figures/boosting_depth.jpeg")
plot(parameter$boosting$depth,parameter$boosting$depthACC,main="10-Fold CV for Boosting",
     xlab="Interaction Depth Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$boosting$depth[which.max(parameter$boosting$depthACC)],max(parameter$boosting$depthACC),
       pch=4,col="red",lwd=3)
dev.off()


## random forests #############################

# mtry
jpeg("./figures/rf_mtry.jpeg")
plot(parameter$randomForest$mtry,parameter$randomForest$mtryACC,main="10-Fold CV for Random Forests",
     xlab="Number of Variables Sampled Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$randomForest$mtry[which.max(parameter$randomForest$mtryACC)],max(parameter$randomForest$mtryACC,na.rm=T),
       pch=4,col="red",lwd=3)
dev.off()

# number of Trees
jpeg("./figures/rf_ntree.jpeg")
plot(parameter$randomForest$ntree,parameter$randomForest$ntreeACC,main="10-Fold CV for Random Forests",
     xlab="Number of Trees Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$randomForest$ntree[which.max(parameter$randomForest$ntreeACC)],max(parameter$randomForest$ntreeACC),
       pch=4,col="red",lwd=3)
dev.off()

# node size
jpeg("./figures/rf_nodesize.jpeg")
plot(parameter$randomForest$nodesize,parameter$randomForest$nodesizeACC,main="10-Fold CV for Random Forests",
     xlab="Node Size Parameter",ylab="Cross Validation Accuracy",type="l")
points(parameter$randomForest$nodesize[which.max(parameter$randomForest$nodesizeACC)],max(parameter$randomForest$nodesizeACC),
       pch=4,col="red",lwd=3)
dev.off()

























