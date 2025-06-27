Sys.setenv(LANGUAGE='en') 
# LOADING THE ADVERTISING DATASET

Advertising <- read.csv("Advertising.csv")
n <- dim(Advertising)[1]

# note that 
# 1. the data.frame has not been attached
# 2. the first column is has not been removed


# function names(): get the variable names

var.names <- names(Advertising)
var.names


# DIFFERENT WAYS OF USING THE FUNCITON lm()
############################################

# this command produces an error 
mod.out <- lm(sales~TV+radio+newspaper)

# we need either to attach() the data.frame or 
# to use the argument "data"

mod.out <- lm(sales~TV+radio+newspaper, data=Advertising)
summary(mod.out)

# the use of the argument "data" allows us to use the
# following syntax

mod.out <- lm(sales~., data=Advertising)
summary(mod.out)

# the dot stands for "all the remaining variables in the data.frame",
# but this also includes the variable X, that makes no sense in this case. 
# We can remove it as:

mod.out <- lm(sales~.-X, data=Advertising)
summary(mod.out)

# let's take a look at the variable TV

hist(Advertising$TV, breaks = seq(0, 300, by=50), prob=TRUE)
sum(Advertising$TV<50 | Advertising$TV>250)

# we want remove from the Adverting dataset 
#
# 1. the observations with TV budget less than 50 and greater than 250
# 2. the variable X
# 
# There exist alternative, equivalent methods to do that, 

# Method 1: logical vector
#-------------------------

is.in.dataset   <- Advertising$TV>=50 & Advertising$TV<=250 
is.in.dataset[1:10]
Advertising.new <- Advertising[is.in.dataset, -1] 

# check dimensions
dim(Advertising)
dim(Advertising.new)


# Method 2: vector of indexes 
#----------------------------

idx.in.dataset <- (1:200)[Advertising$TV>=50 & Advertising$TV<=250]
Advertising.new <- Advertising[idx.in.dataset, -1] 

# check dimension
dim(Advertising.new)

# Method 3: function subset()
#------------------------------

Advertising.new <- subset(Advertising, subset=(Advertising$TV>=50 & Advertising$TV<=250), select=-1)
Advertising.new <- subset(Advertising, subset=is.in.dataset, select=-1)

# Apply lm() to a subset of the data
#-------------------------------------
  
subset.out <- lm(sales~., data=Advertising.new)
summary(subset.out)

# alternative ways to do the same, directly with lm()

subset.out <- lm(sales~.-X, subset = is.in.dataset, data=Advertising)
summary(subset.out)

subset.out <- lm(sales~.-X, subset = idx.in.dataset, data=Advertising)
summary(subset.out)


# Fit lm with all predictors 

mod.full <- lm(sales~.-X, data=Advertising)
summary(mod.full)

# compute explicitly Multiple R-square value

e   <- residuals(mod.out)
RSS <- sum(e^2)
TSS <- var(Advertising$sales)*(n-1)

R.square <- 1-RSS/TSS
R.square

# use update() to modify the model

mod.red <- update(mod.full, .~.-newspaper)
summary(mod.red)

# use the F-test to compare the full and the reduced model

e0 <- residuals(mod.red)
RSS0 <- sum(e0^2)

F.obs <- (RSS0-RSS)/(RSS/(n-4))
F.obs

p.value <- 1-pf(F.obs, df1=1, df2=n-4)
p.value

# verify that F.obs is the square to the observed value of the t.test

summary(mod.full)
(-0.177)^2
