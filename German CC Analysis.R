### Objective of analysis.. Minimization of risk and maximization of profit on behalf of the bank.


### This Credit Data contains data on 20 variables and the classification whether an applicant is considered a Good or a Bad credit risk for 1000 loan applicants

## Loading the data
url="http://freakonometrics.free.fr/german_credit.csv"
credit=read.csv(url, header = TRUE, sep = ",")
View(credit)

colnames(credit)

str(credit)

#################  EDA & Data Pre-Processing


## table plots
## Creating Marginal Proportion Table...

library(gmodels)
attach(credit)

margin.table(prop.table(table(Duration.in.Current.address, Most.valuable.available.asset, Concurrent.Credits,No.of.Credits.at.this.Bank,Occupation,No.of.dependents,Telephone, Foreign.Worker)),1)

margin.table(prop.table(table(Duration.in.Current.address, Most.valuable.available.asset, Concurrent.Credits,No.of.Credits.at.this.Bank,Occupation,No.of.dependents,Telephone, Foreign.Worker)),2)

margin.table(prop.table(table(Duration.in.Current.address, Most.valuable.available.asset, Concurrent.Credits,No.of.Credits.at.this.Bank,Occupation,No.of.dependents,Telephone, Foreign.Worker)),3)

margin.table(prop.table(table(Duration.in.Current.address, Most.valuable.available.asset, Concurrent.Credits,No.of.Credits.at.this.Bank,Occupation,No.of.dependents,Telephone, Foreign.Worker)),4)


## let's build few K1 X K2 contingency table

CrossTable(Creditability, Account.Balance, digits=1, prop.r=F, prop.t=F, prop.chisq=F, chisq=T)

CrossTable(Creditability, Payment.Status.of.Previous.Credit, digits=1, prop.r=F, prop.t=F, prop.chisq=F, chisq=T)

CrossTable(Creditability, Purpose, digits=1, prop.r=F, prop.t=F, prop.chisq=F, chisq=T)

CrossTable(Creditability, Guarantors, digits=1, prop.r=F, prop.t=F, prop.chisq=F, chisq=T)


### Descriptive Statistics for Summarising continous variable (Age, CreditAmount, Duration of credit)

attach(credit)

summary(Duration.of.Credit..month.)

binsCredit <- seq(0, 80, 10)

hist(Duration.of.Credit..month., breaks = binsCredit, xlab = "credit month", ylab = "frequency", main = " ", cex=0.4)

boxplot(Duration.of.Credit..month., bty="n",xlab = "Credit Month", cex=0.4) # For boxplot

## for age
summary(Age..years.)
binsAge <- seq(0, 80, 5)

hist(Age..years., breaks = binsAge, xlab = "Age", ylab = "frequency", main = " ", cex=0.4)

boxplot(Age..years., bty="n",xlab = "Age", cex=0.4)

## credit amount
summary(Credit.Amount)
binsCA <- seq(0, 20000, 1000) # third value specifies bin-width

hist(Credit.Amount, breaks = binsCA, xlab = "credit", ylab = "frequency", main = " ", cex=1)

boxplot(Credit.Amount, bty="n",xlab = "credit", cex=0.4)



#In preparation of predictors to use in building a logistic regression model, we consider bivariate association of the response (Creditability) with the categorical predictors




#'data.frame':	1000 obs. of  21 variables

F=c(1,2,4,5,7,8,9,10,11,12,13,15,16,17,18,19,20)
for(i in F) credit[,i]=as.factor(credit[,i])

training <- sample(1:nrow(credit),size=666)
testing <- sample(1:nrow(credit))[-training]




##########  Logistic Regression


attach(credit)
# THe first model we can fit is a logistic regression, on selected covariates.
model <- glm(Creditability~Account.Balance + Credit.Amount +
               Sex...Marital.Status + Age..years. + Concurrent.Credits, family = binomial,
             data = credit[training,])


summary(model)

# Based on the model it is possible to draw the ROC curve and to compute the AUC

fitLog <- predict(model, type = "response", newdata = credit[testing,] )

library(ROCR)

pred = prediction(fitLog, credit$Creditability[testing])
perf <- performance(pred, "tpr", "fpr")
plot(perf)
AUCLog1=performance(pred, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCLog1,"n")

###### Alternative approach here is to consider logistic reg. on all explanatory variable

LogisticModel <- glm(Creditability~. , family = "binomial" , data = credit[training,])

fitLog <- predict(LogisticModel, type = "response", newdata = credit[testing,] )

pred = prediction(fitLog, credit$Creditability[testing])
perf <- performance(pred, "tpr", "fpr")
plot(perf)
AUCLog1=performance(pred, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCLog1,"n")

## there is slight improvement here from the previous model




#########  Regression Tree based method





#### now consider regression tree (on all covariates)

library(MASS)
library(rpart)
library(tree)

treeModel <- tree(Creditability ~ ., 
                     data = credit[training,])

summary(treeModel)
plot(treeModel)
text(treeModel, pretty = 0, cex = 0.6)


tree_pred <- predict(treeModel, credit[testing,], type = "class")

table(tree_pred, credit$Creditability[testing])
#accuracy = 77.2%

validated_tree = cv.tree(treeModel)
plot(validated_tree$size,
     validated_tree$dev,
     type = "b",
     ylab = "RSS",
     xlab = "Size of the Tree")


tree_prune <- prune.misclass(treeModel, best = 8)

tree_prune_pred <- predict(tree_prune, credit[testing,], type = "class")
table(tree_prune_pred, credit$Creditability[testing])
# accuracy = 77.24%



### ROC curve for that model

library(rpart)
ArbreModel <- rpart(Creditability ~ ., 
                        data = credit[training,])
#We can visualize the tree using

library(rpart.plot)
prp(ArbreModel,type=2,extra=1)


fitArbre <- predict(ArbreModel,
                          newdata=credit[testing,],type="class")[,2]

pred = prediction( fitArbre, credit$Creditability[testing])
perf <- performance(pred, "tpr", "fpr")
plot(perf)
AUCArbre=performance(pred, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCArbre,"n")



################ Discriminant Analysis

#it is M.learning to find a linear combination of features that characterizes or separates two or more classes of objects or events



#For discriminant analysis all the predictors are not used. Only the continuous variables and the ordinal variables are used as for the nominal variables there will be no concept of group means and linear discriminants will be difficult to interpret. The predictors are assumed to have a multivariate normal distribution

# difference between LDA and PCA
## PCA is an unsupervised learning technique (don't use class information) while LDA is a supervised technique (uses class information), but both provide the possibility of dimensionality reduction, which is very useful for visualization


library(MASS)
## linear discriminant analysis

ldafit <- lda(Creditability ~ Value.Savings.Stocks + Length.of.current.employment + Duration.of.Credit..month.+ Credit.Amount + Age..years., subset = training)

ldafit            # gives coefficients of linear discriminants
plot(ldafit)

lda.pred <- predict(ldafit, newdata = credit[testing,])
ldaclass <- lda.pred$class

table(ldaclass, credit$Creditability[testing])
# accuracy = (221+24)/334 = 73.35 %


## Quadratic discriminant analysis

qdafit <- qda(Creditability ~ Value.Savings.Stocks + Length.of.current.employment + Duration.of.Credit..month.+ Credit.Amount + Age..years., subset = training)

qdafit
plot(qdafit)

qda.pred <- predict(qdafit, newdata = credit[testing,])
qdaclass <- qda.pred$class

table(qdaclass, credit$Creditability[testing])
# accurracy = (185 + 54)/334 = 71.5 %


## Neither logistic regression nor discriminant analysis is performing well for this data. The reason DA may not do well is that, most of the predictors are categorical and nominal predictors are not used in this analysis.





#############    Lasso Regression  #####




library(glmnet)

mat1 <- model.matrix(Creditability ~ . , data = credit  )
lassoModel <- glmnet(y= as.numeric(Creditability), x = mat1 )


lassoModel <- glmnet(y= as.numeric(Creditability), x = mat1, family = "binomial" )#, type.measure)

## if you have two predictors that are perfectly collinear, the lasso will pick one of them essentially at random to get the full weight and the other one will get zero weight


# plotting lasso variable selection procedure
op <- par(mfrow=c(1, 2))
plot(lassoModel)
plot(lassoModel, "lambda", label = TRUE)

######### interpretaion

#In both plots, each colored line represents the value taken by a different coefficient in your model. Lambda is the weight given to the regularization term (the L1 norm), so as lambda approaches zero, the loss function of your model approaches the OLS loss function. Here's one way you could specify the LASSO loss function to make this concrete:
# 
# ??lasso=argmin [RSS(??)+?????L1-Norm(??)]
# ??lasso=argmin [RSS(??)+?????L1-Norm(??)]
# Therefore, when lambda is very small, the LASSO solution should be very close to the OLS solution, and all of your coefficients are in the model. As lambda grows, the regularization term has greater effect and you will see fewer variables in your model (because more and more coefficients will be zero valued).
# 
# As I mentioned above, the L1 norm is the regularization term for LASSO. Perhaps a better way to look at it is that the x-axis is the maximum permissible value the L1 norm can take. So when you have a small L1 norm, you have a lot of regularization. Therefore, an L1 norm of zero gives an empty model, and as you increase the L1 norm, variables will "enter" the model as their coefficients take non-zero values.
# 
# The plot on the left and the plot on the right are basically showing you the same thing, just on different scales




## predicting in terms of coefficients and response.

lasso_pred <- predict(lassoModel, credit[testing,], type="coefficients")

## for finding out the frequency
xtest <- mat1[testing,]
ytest <- credit$Creditability[testing]

lasso_pred <- predict(lassoModel, newx = xtest , type="response", s=0.01)
result <- table(ytest, floor(lasso_pred + 1.5))  # once check for this prediction technique.
result

# accuracy = 78.74%





#########  Ensemble Method ( Random Forest)   ################



##As expected, a single tree has a lower performance, compared with a logistic regression. And a natural idea is to grow several trees using some boostrap procedure, and then to agregate those predictions.


library(randomForest)
RF <- randomForest(Creditability ~ .,
                     data = credit[training,], ntree=200, importance=T, proximity=T)

plot(RF, main ="")  # error plot
RF

RF_pred <- predict(RF, credit[testing,], type = "class")

table(RF_pred, credit$Creditability[testing])
# accuracy = 91.3%

importance(RF)

# Importance of predictors are given in the following dotplot.
varImpPlot(RF,  main="", cex=0.8)


## ROC curve analysis
fitForet <- predict(RF,
                      newdata=credit[testing,],
                      type="prob")[,2]
pred = prediction( fitForet, credit$Creditability[testing])
perf <- performance(pred, "tpr", "fpr")
plot(perf)
AUCRF=performance(pred, measure = "auc")@y.values[[1]]
cat("AUC: ",AUCRF,"n")







### Here this model is (slightly) better than the logistic regression. Actually, if we create many training/validation samples, and compare the AUC, we can observe that - on average - random forests perform better than logistic regressions,


AUC=function(i){
     set.seed(i)
     i_test=sample(1:nrow(credit),size=333)
     i_calibration=(1:nrow(credit))[-i_test]
     LogisticModel <- glm(Creditability ~ ., 
                               family=binomial, 
                               data = credit[i_calibration,])
     summary(LogisticModel)
     fitLog <- predict(LogisticModel,type="response",
                        newdata=credit[i_test,])
     library(ROCR)
     pred = prediction( fitLog, credit$Creditability[i_test])
     AUCLog2=performance(pred, measure = "auc")@y.values[[1]] 
     RF <- randomForest(Creditability ~ .,data = credit[i_calibration,])
     fitForet <- predict(RF,
                          newdata=credit[i_test,],
                          type="prob")[,2]
     pred = prediction( fitForet, credit$Creditability[i_test])
     AUCRF=performance(pred, measure = "auc")@y.values[[1]]
     return(c(AUCLog2,AUCRF))
   }
A=Vectorize(AUC)(1:50)
plot(t(A))



