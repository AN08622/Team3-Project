rm(list=ls())

library(MASS)
data(Boston)
attach(Boston)

# We will now try to predict per capita crime rate in the Boston data
# set.
# (a) Try out some of the regression methods explored in this chapter,
# such as best subset selection, the lasso, ridge regression, and
# PCR. Present and discuss results for the approaches that you
# consider.




#splitting data to training and tests set
set.seed(333)

train <- sample(1:nrow(Boston), nrow(Boston)*0.80)

######Best subset selection
library(leaps)
best_subset <- regsubsets(crim ~., data=Boston[train,], nvmax=13)
summary(best_subset)

#using validation set approach to pick the best model
test_matrix <- model.matrix(crim~., data=Boston[-train,])

val.errors <- rep(NA,13)
for (i in 1:13) {
  coefi <- coef(best_subset,id=i)
  pred <- test_matrix[,names(coefi)] %*% coefi
  val.errors[i] <- mean((Boston$crim[-train]-pred)^2)
}

which.min(val.errors)

plot(val.errors,type= 'b')
coef(best_subset, 9)
(best_subset_mse <- val.errors[2])

##########Choose the model using cross-validation with best subset method
predict.regsubsets <- function (object, newdata, id,...){
  form<-as.formula(object$call [[2]])
  mat<-model.matrix(form,newdata)
  coefi<-coef(object, id=id)
  xvars<-names(coefi)
  mat[,xvars]%*%coefi
}

k <- 10
set.seed(333)
folds <- sample(1:k,nrow(Boston), replace=TRUE)
cv.errors <- matrix(NA,k,13, dimnames=list(NULL, paste(1:13)))

for(j in 1:k){
  best.fit=regsubsets(crim~., data=Boston[folds!=j, ], nvmax=13)
  for(i in 1:13) {
    pred=predict(best.fit,Boston[folds==j,],id=i)
    cv.errors[j,i]=mean((Boston$crim[folds==j]-pred)^2)
  }
}

(mean.cv.errors <- apply(cv.errors,2,mean))
plot(mean.cv.errors, type="b")
which.min(mean.cv.errors)
#lowest is at 12 but we can use 9 instead as the
#errors are quite similar and we are parsimonious
(best_subset_mse_cv <- mean.cv.errors[9])
#43.6168

#######Lasso Regression
library(glmnet)
set.seed(333)

x <- model.matrix(crim~., Boston)[, -1]
y <- Boston$crim

lasso.fit <- glmnet(x[train, ], y[train], alpha=1)
cv.lasso.fit <- cv.glmnet(x[train, ], y[train], alpha=1)
plot(cv.lasso.fit)
(bestlam.lasso <- cv.lasso.fit$lambda.min)
#0.05101479

lasso.pred <- predict(lasso.fit, s=bestlam.lasso, newx=x[-train,]) 
(lasso.error <- mean((lasso.pred-y[-train])^2))
#lasso mse = 68.51329



########Ridge Regression
ridge.fit <- glmnet(x[train, ], y[train], alpha=0)
cv.ridge.fit <- cv.glmnet(x[train, ], y[train], alpha=0)
plot(cv.ridge.fit)

(bestlam.ridge <- cv.ridge.fit$lambda.min)
#0.5344389

ridge.pred <- predict(ridge.fit, s=bestlam.ridge, newx=x[-train,]) 
(ridge.error <- mean((ridge.pred-y[-train])^2))
#lasso mse = 69.43538



###PCR
install.packages("pls")
library(pls)
set.seed(333)

pcr.fit <- pcr(crim~., data=Boston, subset=train, scale=TRUE, validation ="CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type="MSEP")
#lowest MSE at 13. But we see that it is very similar to 8-13.
#Hence for parsinmony, we pick 8 variables

pcr.pred <- predict(pcr.fit, x[-train,], ncomp=13) 
y.test <- crim[-train]
(pcr.error <- mean((pcr.pred-y.test)^2))
#PCR MSE = 68.20879


# (b) Propose a model (or set of models) that seem to perform well on
# this data set, and justify your answer. Make sure that you are
# evaluating model performance using validation set error, crossvalidation, or some other reasonable alternative, as opposed to
# using training error.


#########Model to see the best subset
errors <- c(best_subset_mse, lasso.error, ridge.error, pcr.error)
names(errors) <- c("best_subset_mse", "lasso.error", "ridge.error", "pcr.error")
barplot(sort(errors, decreasing = F))
#Picking lowest mse model: which is best subset MSE


# (c) Does your chosen model involve all of the features in the data
# set? Why or why not?

########Best Subset Mode info

#MSE: 67.21199

#No, The dataset had 12 variables (excluding crime) and we have 9 variables
#as mentioned below

# (Intercept)          zn       indus         nox         dis         rad 
# 17.91831317  0.04007497 -0.10561950 -8.58235521 -0.84508484  0.49883640 
# ptratio       black       lstat        medv 
# -0.22308937 -0.01410997  0.15806233 -0.14477712  








