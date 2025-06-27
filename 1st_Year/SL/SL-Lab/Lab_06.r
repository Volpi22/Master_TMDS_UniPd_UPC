###############################
# LOGISTIC CLASSIFIER
##############################

library(ISLR2)
data("Default")
attach(Default)

# scatterplot of the data with the two classes

# obtain a "0/1" response variable 
default01 <- as.numeric(default=="Yes")
plot(balance, income, col=default01+1, pch=default01*15+1)

#########################
# logistic classifier
#########################

# check that 1="YES" and 0="NO"
contrasts(default)

# fit the logistic regression model
glm.fit <- glm(default ~ income + balance, family = binomial)

# obtain the estimated probabilities 
glm.prob <- predict(glm.fit, type="response")

# assign probabilities>0.5 to the class "YES"
glm.class <- rep("No", 10000)
glm.class[glm.prob>0.5] <- "Yes"


# decision boundary

summary(glm.fit)
beta.hat <- coefficients(glm.fit)
beta0.hat <- beta.hat[1]
beta1.hat <- beta.hat[2]
beta2.hat <- beta.hat[3]


intercept <- -beta0.hat/beta1.hat
slope <- -beta2.hat/beta1.hat

plot(balance, income, col=default01+1, pch=default01*15+1)
abline(intercept, slope, lwd=2, col="blue")

# confusion matrix

conf.matrix <- table(glm.class, default)
conf.matrix

# overall (training) error rate
(conf.matrix[1,2]+conf.matrix[2,1])/sum(conf.matrix)

(225+38)/10000

# trivial classifier that assigns "No" to all the 
# units

trivial.pred <- rep("No", 10000)

# confusion matrix for the trivial predictor
table(trivial.pred, default)

# overall (training) error rate for the trivial classifier
333/10000

# (training) error rate among individuals who defaulted 
conf.matrix
conf.matrix[1,2]/sum(conf.matrix[1:2, 2])

225/(225+108)

######################################
#  Different threshold
#
#  threshold=0.2
######################################


glm.class <- rep("No", 10000)
glm.class[glm.prob>0.2] <- "Yes"

conf.matrix <- table(glm.class, default)
conf.matrix

# overall (training) error rate
(conf.matrix[1,2]+conf.matrix[2,1])/sum(conf.matrix)

(133+271)/10000

# (training) error rate among individuals who defaulted 
conf.matrix[1,2]/sum(conf.matrix[1:2, 2])

133/(133+200)

# (training) error rate among individuals who did not default 
conf.matrix[2,1]/sum(conf.matrix[1:2, 1])

271/(271+9396)


####################################
# computing performance measures
####################################


# Confusion matrix with threshold = 0.5

glm.class <- rep("No", 10000)
glm.class[glm.prob>0.2] <- "Yes"

conf.matrix <- table(glm.class, default)
conf.matrix[2:1, 2:1]

n <- sum(conf.matrix) # sample size

# extract relevant quantities from the confusion matrix

# true positive
TP <- conf.matrix["Yes","Yes"]
TP

# true negative
TN <- conf.matrix["No", "No"]
TN

# false positive
FP <- conf.matrix["Yes","No"]
FP

# false negative 
FN <- conf.matrix["No", "Yes"]
FN

# positive
P     <- TP + FN

# negative
N     <- FP + TN

# predicted positive
P.ast <- TP + FP

# compute the performance measures

# overall error rate
OER <- (FP+FN)/n
OER

# Positive Predicted Values (Precision)
PPV <- TP/P.ast
PPV

# True Positive Rate (Sensitivity, Recall) 
# that refers to Pr(positive | positive)
TPR <- TP/P
TPR

# F1 score (harmonic mean of PPV and TPR)
F1  <- 2*PPV*TPR/(PPV+TPR)
F1

# True Negative Rate (Specificity)
# that refers to Pr(negative | negative)
TNR <- TN/N
TNR

# False Positive Rate
FPR <- FP/N
FPR


# Function for the computation of performance measures
#
# Arguments:
#
# true.values = vector of true values
# pred.values = vector of predicted values
# lab.pos     = label of the positive class
#
perf.measure <- function(true.values, pred.values,  lab.pos = 1){
  #
  # compute the confusion matrix and number of units
  conf.matrix <- table(pred.values, true.values)
  n <- sum(conf.matrix)
  #
  # force the label of positives to be a character string
  lab.pos <- as.character(lab.pos)
  #
  # obtain the label of negatives
  lab <- rownames(conf.matrix)
  lab.neg <- lab[lab != lab.pos]
  #
  # extract relevant quantities from the confusion matrix
  TP <- conf.matrix[lab.pos, lab.pos]
  TN <- conf.matrix[lab.neg, lab.neg]
  FP <- conf.matrix[lab.pos, lab.neg]
  FN <- conf.matrix[lab.neg, lab.pos]
  P     <- TP + FN
  N     <- FP + TN
  P.ast <- TP + FP
  #
  # compute the performance measures
  OER <- (FP+FN)/n
  PPV <- TP/P.ast
  TPR <- TP/P
  F1  <- 2*PPV*TPR/(PPV+TPR)
  TNR <- TN/N
  FPR <- FP/N
  return(list(overall.ER = OER, PPV=PPV, TPR=TPR, F1=F1, TNR=TNR, FPR=FPR))
}

PM <- perf.measure(default, glm.class,  lab.pos="Yes")
PM

#################################################
# the ROC (Receiver Operating Characteristic) curve 
#################################################


library(pROC)

# If labels are "0" and "1" then "0"=negative and "1"=positive
# levels = negative (0's) as first element and positive (1's) as second

roc.out <- roc(default, glm.prob, levels=c("No", "Yes"))

# different ways of plotting the ROC curve
plot(roc.out) # check values on the x axis

# legacy.axes=TRUE   1 - specificity on the x axis
plot(roc.out, legacy.axes=TRUE)

# change the labels: 
#  Sensitivity = TPR
#  Specificity = TNR

plot(roc.out, legacy.axes=TRUE, xlab="1 - True Negative Rate", ylab="True Positive Rate")

# change the labels: 
# 1 - TNR = FPR 
plot(roc.out, legacy.axes=TRUE, xlab="False Positive Rate", ylab="True positive rate")

# compute AUC = Area Under the Curve
plot(roc.out,  print.auc=TRUE, legacy.axes=TRUE, xlab="False Positive Rate", ylab="True Positive Rate")
auc(roc.out)

# specificity (TNR) and sensitivity (TPR) for a given threshold
coords(roc.out, 0.5)

coords(roc.out, seq(0.1, 0.9, by=0.1))

# threshold that maximizes the sum of specificity (TNR) and sensitivity (TPR)
coords(roc.out, "best")


######################################
# Linear Discriminant Analysis (LDA)
######################################

library(MASS)
lda.fit <- lda(default~balance+income, data=Default)

# see some summary values 
lda.fit
plot(lda.fit)
plot(lda.fit, type="density")

lda.pred <- predict(lda.fit)
names(lda.pred)

# take a look at the components  of lda.pred

lda.pred$posterior[10:20,]
lda.pred$class[10:20]

# classification is obtained with threshold = 0.5
# but an arbitrary threshold could be speficied as 

thr <- 0.2 # use also 0.2 as in logistic
lda.class <- rep("No", 10000)
lda.class[lda.pred$posterior[,2]>= thr] <- "Yes"

# check that for thr=0.5 lda.class and lda.pred$class are equal
sum(lda.class!=lda.pred$class)

# see the confusion matrix, the performance measures and the ROC curve
conf.matrix <- table(lda.class, default)
conf.matrix

perf.measure(default, lda.class, lab.pos = "Yes")

roc.out <- roc(default, lda.pred$posterior[,2], levels=c("No", "Yes"))
plot(roc.out,  print.auc=TRUE, legacy.axes=TRUE, xlab="False Positive Rate", ylab="True Positive Rate")


#########################################
# Quadratic Discriminant Analysis (QDA)
#########################################

qda.fit <- qda(default~balance+income, data=Default)
qda.fit

qda.pred <- predict(qda.fit)

# also in this case
# classification is obtained with threshold = 0.5
# but an arbitrary threshold could be speficied as 

thr <- 0.2 # use also 0.2 as in logistic
qda.class <- rep("No", 10000)
qda.class[qda.pred$posterior[,2]>= thr] <- "Yes"


# check that for thr=0.5 qda.class and qda.pred$class are equal
sum(qda.class!=qda.pred$class)


conf.matrix <- table(qda.class, default)
conf.matrix


# overall misclassification error rate
perf.measure(default, qda.class, lab.pos = "Yes")

roc.out <- roc(default, qda.pred$posterior[,2], levels=c("No", "Yes"))
plot(roc.out,  print.auc=TRUE, legacy.axes=TRUE, xlab="False Positive Rate", ylab="True Positive Rate")

#########################################
# Naive Bayes
#########################################

library(e1071)

# NaiveBayes classifier assuming independence 
# and normal distribution of predictors

nb.fit <- naiveBayes(default~balance+income, data=Default)

# 
# check the nb.fit object:

# $apirori
#
nb.fit$apriori

# "apriori" class distribution for the dependent variable.

# $tables
#
# A list of tables, one for each predictor variable. 
# For each categorical variable a table giving, 
# for each attribute level, the conditional probabilities 
# given the target class. For each numeric variable, 
# a table giving, for each target class, mean and standard 
# deviation of the (sub-)variable.

nb.fit$tables


# use type = "class" for classification with threshold = 0.5
# note that this is the default
#
nb.posterior <- predict(nb.fit, Default, type="raw")
nb.class     <- predict(nb.fit, Default, type="class") # type="class" can be omitted because it is the dafalt value

# also in this case
# classification is obtained with threshold = 0.5
# but an arbitrary threshold could be speficied as 

thr <- 0.2 # use also 0.2 as in logistic
nb.new.class <- rep("No", 10000)
nb.new.class[nb.posterior[,2]>= thr] <- "Yes"


# check that for thr=0.5 qda.class and qda.pred$class are equal
sum(nb.new.class!=nb.class)


table(nb.new.class, default)

# note that that it is only slightly worse than
# Quadratic Discriminant Analysis
#
perf.measure(default, nb.new.class, lab.pos="Yes")


roc.out <- roc(default, nb.posterior[,2], levels=c("No", "Yes"))
plot(roc.out,  print.auc=TRUE, legacy.axes=TRUE, xlab="False Positive Rate", ylab="True Positive Rate")


#################################
# COMPARE METHODS FOR thr=0.2
#################################


       TPR    PPV   F1
       
GLM    0.60  0.42  0.498

LDA    0.57  0.44  0.496

QDA    0.63  0.40  0.487

NB     0.62  0.39  0.479




