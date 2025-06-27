# THE ADVERTISING DATASET

Advertising <- read.csv("Advertising.csv")
Advertising <- Advertising[, -1]
attach(Advertising)

plot(TV, sales, pch=20)


# sample size
n <- length(sales)
n

###############################
# COVARIANCE AND CORRELATION
###############################

# computation of the covariance between TV and sales

cov.st <- sum((sales-mean(sales))*(TV-mean(TV)))/(n-1)
cov.st
cov(sales, TV)

# HOW TO INTERPRET THE MAGNITUDE OF 
# THE COVARIANCE BETWEEN TWO VARIABLES

# COVARIANCE MATRIX, using a build-in R function

cov(Advertising)
cov.mat <- cov(Advertising)

# compute the correlation matrix from the covariance matrix 
# by scaling to obtain diagonal entries equal to one

D <- diag(cov.mat)
D 
D <- diag(D)
D

cor.mat <- solve(sqrt(D))%*%cov.mat%*%solve(sqrt(D))
cor.mat

# this is also done by the function cov2cor()
cov2cor(cov.mat)

# some ways to visualize a correlation matrix

library(corrplot)

corrplot(cor(Advertising), method = 'number')
corrplot(cor(Advertising), method = 'color')
corrplot(cor(Advertising), method = 'ellipse')

# FUNCTION "pairs" for matrix plot

pairs(Advertising)


###############################
# SIMPLE LINEAR REGRESSION
###############################


x <- TV
y <- sales

plot(x, y, pch=20)

# regression coefficients

beta1.hat <- cov(x, y)/var(x)
beta0.hat <- mean(y)- beta1.hat*mean(x)

beta0.hat
beta1.hat
abline(beta0.hat, beta1.hat, col="blue", lwd=2)

# fitted values

y.hat <- beta0.hat+beta1.hat*x 

points(x, y.hat, col="red", pch=20)

# residuals

e <- y -y.hat
sum(e)

# RSS

RSS <- sum(e^2)
RSS

# RSE

RSE <- sqrt(RSS/(200-2))
RSE

# the lm() function

reg.out <- lm(y~x)
summary(reg.out)

# extraction of useful quantities

e <- residuals(reg.out)
e[1:10]

y.hat <- fitted.values(reg.out)
y.hat[1:10]

beta.hat <- coefficients(reg.out)
beta.hat


# add regression line to the plot

plot(x, y, pch=16)
abline(beta.hat[1], beta.hat[2])

# more compactly
abline(reg.out)

# confidence intervals for the parameters

confint(reg.out)
confint(reg.out, level=0.8)


# checking properties of the residuals

e     <- residuals(reg.out)
y.hat <- fitted.values(reg.out)
sum(e)
sum(e*x)
sum(e*y.hat)

# Decomposition of variability and
# R-squared statistic

TSS <- sum((y-mean(y))^2)
RSS <- sum((y-y.hat)^2)

R2 <- (TSS-RSS)/TSS
R2

