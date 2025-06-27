######################
# CROSS VALIDATION
######################

library(ISLR2)
data(Auto)
dim(Auto)
n <- dim(Auto)[1]

# regression of mpg on horsepower
plot(Auto$horsepower, Auto$mpg, xlab="horsepower", ylab="mpg", pch=20)

# linear model (degree 1)
##############################

lm.fit <- lm(mpg~horsepower, data=Auto)
summary(lm.fit)
abline(lm.fit, lwd=2, col="blue")

# compute the mean squared error 
MSE <- mean(residuals(lm.fit)^2)
MSE


# polynomial with degree 2
##############################

lm.fit <- lm(mpg~poly(horsepower, degree=2), data=Auto)
#summary(lm.fit)

# add the curve to the plot
x <- seq(min(Auto$horsepower), max(Auto$horsepower), length=1000)
new.x <- data.frame(horsepower=x)
y.hat <- predict(lm.fit, new.x)
lines(x, y.hat, lwd=2, col="green")

# compute the mean squared error 
MSE <- mean(residuals(lm.fit)^2)
MSE



# polynomial with degree 10
##############################

lm.fit <- lm(mpg~poly(horsepower, degree=10), data=Auto)
#summary(lm.fit)

# add the curve to the plot
y.hat <- predict(lm.fit, new.x)
lines(x, y.hat, lwd=2, col="red")

# compute the mean squared error 
MSE <- mean(residuals(lm.fit)^2)
MSE

# Compute the training-MSE for all the  
# polynomials of degree from 1 to 10
#
MSE.train <- rep(0, 10)
for(i in 1:10){
  lm.fit <- lm(mpg~poly(horsepower, degree=i), data=Auto)
  MSE.train[i] <- mean(residuals(lm.fit)^2)
}

MSE.train
which.min(MSE.train)
plot(MSE.train, type="b", pch=16, ylim=c(17,25), xlab="degree of polynomial")

################################
# the validation set approach
################################

# randomly split the dataset into 
# a train set and a validation set
# of the same size (n/2)

dim(Auto)
n/2
set.seed(100)
train <- sample(1:392, size=196, replace= FALSE)
Auto.train <- Auto[ train,]
Auto.val   <- Auto[-train,]

# fit the linear model on the training data
lm.fit <- lm(mpg~horsepower, data=Auto.train)

# predict with the validation set 
y.pred <- predict(lm.fit, newdata=Auto.val)

# Compute the Mean Squared Error on the validation set
#
MSE <- mean((Auto.val$mpg-y.pred)^2)
MSE 

# Compute the validation-MSE for all the  
# polynomials of degree from 1 to 10
#
#
MSE.val   <- rep(0, 10)
for(i in 1:10){
  lm.fit <- lm(mpg~poly(horsepower, degree=i), data=Auto.train)
  y.pred <- predict(lm.fit, newdata=Auto.val)
  MSE.val[i] <- mean((Auto.val$mpg-y.pred)^2)
}

# compare the two MSEs (validation vs training)

which.min(MSE.val)

plot(1:10, MSE.val, type="b", ylim=c(17,25), pch=16, xlab="degree", ylab="MSE")
lines(1:10, MSE.train, type="b", col="blue", pch=16)
legend("topright", lty=1, pch=16,  legend =c("validation MSE", "training MSE"), col=c("black", "blue"))

#####################################
# Leave-One-Out Cross-Validation
#####################################

# We are going to use the package "boot"
# that implements functions for cross-validation
# for GLMs, and, for this reason we need to fit linear 
# regression models with the glm function, as follows

glm.fit <- glm(mpg~horsepower, data=Auto, family=gaussian)
coef(glm.fit)

# note that "gaussian" is the default 
# value for the argument "family"
#
glm.fit <- glm(mpg~horsepower, data=Auto)


# glm.fit and lm.fit are different R objects, but 
# they both contain the same fitted linear model

# We can now use the cv.glm() function 
# from the package "boot"
#
library(boot)

# K = n means leave-one-out CV and this is 
# the default value of K
#
cv.err <- cv.glm(Auto, glm.fit, K=392)
cv.err <- cv.glm(Auto, glm.fit)

# check the object cv.err

# $k
# is the number of groups 
cv.err$K

# $delta 
# is a vector of length two. The first component is the raw 
# cross-validation estimate of the prediction error. 
# The second component is the adjusted cross-validation estimate. 
# The adjustment is designed to compensate for the bias introduced 
# by not using all the observations. 

cv.err$delta

# Compute the leave-one-out-CV-MSE for all the  
# polynomials of degree from 1 to 10
#
#
cv.error.loo <- rep(0,10)
start.time <- Sys.time()
for (i in 1:10){
  glm.fit <- glm(mpg~poly(horsepower,i),data=Auto)
  cv.error.loo[i] <- cv.glm(Auto,glm.fit)$delta[1]
}
end.time <- Sys.time()

# execution time
exec.time <-end.time-start.time
exec.time
# average number of models per second
(n*10)/as.numeric(exec.time)

plot(1:10, cv.error.loo, type="b", pch=16, ylim=c(17,25), xlab="degree", ylab="LOOCV error")


################################
# k-Fold Cross-Validation
###############################

# Compute the 10-fold-CV-MSE for all the  
# polynomials of degree from 1 to 10
#
#
set.seed(17)
cv.error.10 <- rep(0,10)
start.time <- Sys.time()
for (i in 1:10){
  glm.fit <- glm(mpg~poly(horsepower,i, raw=TRUE),data=Auto)
  cv.error.10[i] <- cv.glm(Auto,glm.fit,K=10)$delta[1]
}
end.time <- Sys.time()

# execution time
exec.time <-end.time-start.time
exec.time

cv.error.10


plot(1:10, cv.error.loo, type="b", pch=16, ylim=c(17,25), xlab="degree", ylab="", col="blue")
lines(1:10, cv.error.10, type="b", pch=16, ylim=c(17,25), col="red")

legend("topright", lty=1, pch=16,  legend =c("LOOCV error", "10-fold CV error"), col=c("blue", "red"))


