Sys.setenv(LANGUAGE='en') 

####################################################
# EXAMPLE: Auto dataset and selection of variables
####################################################

# package from the book "Introduction to Statistical Learning" 2nd edition
library(ISLR2)

data("Auto")
Auto$origin <- factor(Auto$origin, levels=1:3,  labels=c("American", "European", "Japanese"))

# "name" is not a variable and in the for the moment we do not
# consider categorical variables such as "origin". 

full.mod <- lm(mpg~.-name-origin, data=Auto)
summary(full.mod)

# backward stepwise procedure
red.mod1 <- update(full.mod, ~.-horsepower)
summary(red.mod1)

red.mod2 <- update(red.mod1, ~.-cylinders)
summary(red.mod2)

red.mod3 <- update(red.mod2, ~.-displacement)
summary(red.mod3)

red.mod4 <- update(red.mod3, ~.-acceleration)
summary(red.mod4)

anova(red.mod4, full.mod)


#################
# PREDICTION 
#################


# THE ADVERTISING DATASET

Advertising <- read.csv("Advertising.csv")
attach(Advertising)

plot(TV, sales, pch=20)


# sample size
n <- length(sales)
n

plot(TV, sales, pch=20)
mod.out <- lm(sales~TV)
summary(mod.out)

abline(mod.out, lwd=2)

# use the regression line to predict the 
# response for TV = 170

coefficients(mod.out)

7.03259355+0.04753664*170 

points(170, 15.114, pch="X", col="red")

new.x <- data.frame(TV=170)
predict(mod.out, newdata=new.x)

predict(mod.out, newdata=new.x, interval ="confidence")
predict(mod.out, newdata=new.x, interval ="prediction")


###################################
# DIAGNOSTICS AND RESIDUAL PLOTS
###################################

plot(TV, sales, pch=20)
mod.out <- lm(sales~TV)
summary(mod.out)

# residual plot with covariate "TV" on the x-axis 
#
par(mfrow=c(1,2))
plot(TV, sales)
abline(mod.out, col="blue", lwd=2)
plot(TV, residuals(mod.out), col="gray40", xlab="TV", ylab="residuals")
lines(loess.smooth(TV, residuals(mod.out)), col="blue", lwd=2)
abline(h=0, lty=2)
par(mfrow=c(1,1))

# residual plot with fitted.values on the x-axis 
#
par(mfrow=c(1,2))
plot(TV, sales)
abline(mod.out, col="blue", lwd=2)
plot(fitted(mod.out), residuals(mod.out), col="gray40", xlab="fitted values", ylab="residuals")
lines(loess.smooth(fitted(mod.out), residuals(mod.out)), col="blue", lwd=2)
abline(h=0, lty=2)
par(mfrow=c(1,1))

# residual plots in R

# some residual plots
plot(mod.out)

# the four plots on the same window
par(mfrow=c(2,2))
plot(mod.out)
par(mfrow=c(1,1))

# choose which you want to plot (which in 1:6)

plot(mod.out, which=1)
plot(mod.out, which=3)


#
# AUTO DATASET
#

library(ISLR2)
data(Auto)
attach(Auto)
mod.out <- lm(mpg~horsepower)

summary(mod.out)


# try to use this model for prediction 

plot(horsepower, mpg, pch=16, ylim=c(0, 50))
abline(mod.out, lwd=2, col="blue")
xp <- c(210)
new.x <- data.frame(horsepower=xp)
new.y <- predict(mod.out, newdata=new.x)
points(xp, new.y, pch="X", col="red", cex=1.4)


# check the residual plot

par(mfrow=c(1,2))
plot(horsepower, mpg, ylim=c(0, 50))
abline(mod.out, col="blue", lwd=2)
plot(mod.out, which=1, lwd=2)
par(mfrow=c(1,1))

# polynomial regression fit a
# polynomial of degree 2 instead 
# than a line (that is a polynomial of degree 1) 

mod.out2 <- lm(mpg  ~ horsepower + I(horsepower^2))
summary(mod.out2)

# plot the fitted model 

plot(horsepower, mpg, pch=20, ylim=c(0, 50))
range(horsepower)
xp <- seq(46, 230, length=100)
new.x <- data.frame(horsepower=xp) 
yp <- predict(mod.out2, newdata=new.x)
lines(xp, yp, lwd=2, col="blue")

# check the prediction 

xp <- c(210)
new.x <- data.frame(horsepower=xp)
new.y <- predict(mod.out2, newdata=new.x)
points(xp, new.y, pch="X", col="red", cex=1.4)

# residual plot

plot(mod.out2, which=1, lwd=2)

# function poly() for more general polynomial regression
# Example: polynomial of degree 5

mod.out.poly <- lm(mpg~poly(horsepower, 5))
summary(mod.out.poly)

range(horsepower)
xp <- seq(46, 230, length=100)
new.x <- data.frame(horsepower=xp) 
yp <- predict(mod.out.poly, newdata=new.x)
plot(horsepower, mpg, pch=20)
lines(xp, yp, lwd=2, col="blue")

# residual plot

plot(mod.out.poly, which=1, lwd=2)


##################################
# EXAMPLE OF POLYNOMIAL REGRESSION 
# mcycle dataset
# use of the function poly()
##################################

library(MASS)
data(mcycle)
attach(mcycle)


####

plot(times, accel, xlab="Time", ylab="Acceleration", pch=20)

range(times)
xp <- seq(3, 55, length=100)
x.new <- data.frame(times=xp)

# polynomial regression with the poly() function
# degree=1 (linear regression)

pol1 <- lm(accel~poly(times, degree = 1, raw=FALSE))
summary(pol1)
y.hat <- predict(pol1)
plot(times, accel, xlab="Time", ylab="Acceleration", pch=20)
lines(times, y.hat, col="blue", lwd=2)

# residual plot
plot(pol1, which=1, lwd=2)

# degree=5

pol5 <- lm(accel~poly(times, degree = 5))
summary(pol5)
y.hat <- predict(pol5)
plot(times, accel, xlab="Time", ylab="Acceleration", pch=20)
lines(times, y.hat, col="blue", lwd=2)

# residual plot

plot(pol5, which=1, lwd=2)

# degree=10

pol10 <- lm(accel~poly(times, degree = 10))
summary(pol10)
y.hat <- predict(pol10)
plot(times, accel, xlab="Time", ylab="Acceleration", pch=20)
lines(times, y.hat, col="blue", lwd=2)

# residual plot
plot(pol10, which=1, lwd=2)

# degree=12
pol12 <- lm(accel~poly(times, degree = 12))
summary(pol12)
y.hat <- predict(pol12)
plot(times, accel, xlab="Time", ylab="Acceleration", pch=20)
lines(times, y.hat, col="blue", lwd=2)

# residual plot
plot(pol12, which=1, lwd=2)


###################################
# LOG-TRANSFORM OF RESPONSE
###################################

# an alternative way to deal with non linearity 
# is that of applying transformations to the variables
# such as the log()

plot(horsepower, log(mpg), pch=20)
mod.out3 <- lm(log(mpg)  ~ horsepower)
summary(mod.out3)
abline(mod.out3, lwd=2, col="blue")

plot(mod.out3, which=1, lwd=2)

# plot the fitted curve 

range(horsepower)
xp <- seq(46, 230, length=100)
new.x <- data.frame(horsepower=xp) 
yp <- predict(mod.out3, newdata=new.x)

plot(horsepower, mpg, pch=20)
lines(xp, exp(yp), lwd=2, col="blue")

# prediction with confidence interval 

xp <- c(210)
new.x <- data.frame(horsepower=xp)
log.y.hat <- predict(mod.out3, newdata=new.x, interval="confidence")
log.y.hat
exp(log.y.hat)

#####################################################
# MULTIPLE REGRESSION: LOG-TRANSFORM OF THE RESPONSE
#####################################################

mod.out <- lm(mpg ~. -origin-name, data=Auto)
summary(mod.out)

# residual plot
plot(mod.out, which=1, lwd=2)

# log-transform of the response
mod.out1 <- lm(log(mpg) ~. -origin-name, data=Auto)
summary(mod.out1)

# residual plot
plot(mod.out1, which=1, lwd=2)

# we add a quadratic term for horsepower
mod.out2 <- lm(log(mpg) ~.+horsepower+I(horsepower^2) -origin-name, data=Auto)
summary(mod.out2)

# residual plot
plot(mod.out2, which=1, lwd=2)




