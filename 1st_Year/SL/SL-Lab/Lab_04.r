# THE ADVERTISING DATASET

Advertising <- read.csv("Advertising.csv")
attach(Advertising)


################################
# INTERACTION EFFECTS
################################

mod.out <- lm(sales ~ TV + radio + newspaper)
summary(mod.out)

mod.out <- lm(sales ~ TV + radio)
summary(mod.out)

par(mfrow=c(2,2))
plot(mod.out)
par(mfrow=c(1,1))


mod.out <- lm(sales ~ TV + radio + TV:radio)
summary(mod.out)

par(mfrow=c(2,2))
plot(mod.out)
par(mfrow=c(1,1))

mod.out <- lm(sales ~  TV*radio)


summary(mod.out)


#################################
# CATEGORICAL REGRESSORS
##################################

library(faraway)
data(coagulation)
help(coagulation)

attach(coagulation)


summary(coagulation)
boxplot(coag~ diet, col="cyan", ylab="coagulation")

is.factor(diet)

contrasts(diet)

mod.out <- lm(coag~ diet)
summary(mod.out)

# contrasts and design matrix
diet
contrasts(diet)
model.matrix(mod.out)

# diagnostics

par(mfrow=c(2,2))
plot(mod.out)
par(mfrow=c(1,1))


#########################################
# EXAMPLE: comparing linear regression 
# with a binary covariate with the 
# comparison of means for two populations
#########################################

normtemp <- read.csv("normtemp.txt", sep="", stringsAsFactors = TRUE)
attach(normtemp)
temp.C <- (temperature -32) *5/9 
boxplot(temp.C~gender, col="cyan")

# two-sample t-test

var.test(temp.C~gender)
t.test(temp.C~gender, var.equal=TRUE)

contrasts(gender)

mod.out <- lm(temp.C~gender)
summary(mod.out)

########################
#  ANOVA
#########################

# test for equal variances

bartlett.test(coag~ diet)

aov.diet <- aov(coag~ diet)
summary(aov.diet)

# post-hoc analysis

tukey.diet <- TukeyHSD(aov.diet)
tukey.diet
plot(tukey.diet)

par(mfrow=c(1, 2))
plot(coag~ diet)
plot(tukey.diet)
par(mfrow=c(1,1))


