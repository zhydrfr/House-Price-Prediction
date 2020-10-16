library(corrplot)
library(ggplot2)
library(ggrepel)
library(faraway)
library(nortest) 
library(mice)
library(VIM)
library(lattice)
library(Amelia)


## 1 Outliers
# influence
df_train <- read.csv("C:/Users/lenovo/Downloads/data project/train_pyrr.csv",stringsAsFactors = F)
colnames(df_train)
df_train <- df_train[,-1 ]
mf <- lm(SalePrice ~., data = df_train)
summary(mf)
cook <- cooks.distance(mf) 
cook
length(cook)
halfnorm(cook, nlab = 4, ylab="Cooks distance")
# 0 removal
df_train <- df_train[-c(742,467), ]
# 1st removal
df_train <- df_train[-c(1280,1049,75), ]
# 2nd removal
df_train <- df_train[-c(898,363), ]
# 3rd removal
df_train <- df_train[-c(1061,9), ]
# 4th removal
df_train <- df_train[-c(457), ]
# 5th removal
df_train <- df_train[-c(565), ]
# 6th removal
df_train <- df_train[-c(522,722,560), ]


str(df_train)
dim(df_train)



write.csv(df_train,"C:\\Users\\lenovo\\Downloads\\data project\\train_rpy.csv", row.names = FALSE)




## MultiColinearity
df_train <- read.csv("C:/Users/lenovo/Downloads/data project/train_pyr2.csv",stringsAsFactors = F)
colnames(df_train)
df_train <- df_train[,-1 ]
df_train <- df_train[,-78]
mf <- lm(SalePrice ~., data = df_train)
summary(mf)

# drop Some columns
which(colnames(df_train)=="GrLivArea")
df_train <- df_train[-44]

which(colnames(df_train)=="TotalBsmtSF")
df_train <- df_train[-36]

which(colnames(df_train)=="Exterior1st_Stone")
df_train <- df_train[-78]

# multicolinearity
df_train <- read.csv("C:/Users/lenovo/Downloads/data project/train_pyr2.csv",stringsAsFactors = F)
dim(df_train)
# after dummies
which(colnames(df_train)=="Exterior2nd_CBlock")
which(colnames(df_train)=="Exterior1st_BrkComm")
which(colnames(df_train)=="RoofStyle_Shed")
which(colnames(df_train)=="Condition2_RRNn")
which(colnames(df_train)=="Condition2_PosN")
df_train <- df_train[-c(90,74,66,50,47)]

dim(df_train)

vif(mf)
which.max(vif(mf))
sort(vif(mf), decreasing = TRUE)
which(vif(mf)>60)

# drop highest VIFs
which(colnames(df_train)=="SaleCondition_Normal")
which(colnames(df_train)=="SaleType_WD")
which(colnames(df_train)=="MiscFeature_Shed")
which(colnames(df_train)=="MiscFeature_No")
which(colnames(df_train)=="GarageType_No")
which(colnames(df_train)=="GarageType_Detchd")
which(colnames(df_train)=="GarageType_Attchd")
df_train <- df_train[-c(136,132,123,121,120,119,115)]

which(colnames(df_train)=="Exterior2nd_VinylSd")
which(colnames(df_train)=="Exterior2nd_MetalSd")
which(colnames(df_train)=="Exterior1st_VinylSd")
which(colnames(df_train)=="Exterior1st_MetalSd")
which(colnames(df_train)=="RoofStyle_Hip")
which(colnames(df_train)=="RoofStyle_Gable")
df_train <- df_train[-c(93,88,79,76,62,60)]


# H0 radmishavad YANI hadeaghal yek moteghayere tasir gozar vojood darad
mean(mf$residuals)

# Check residuals
plot(mf$fitted.values , mf$residuals, xlab = "fitted values", ylab="Residuals")
abline(h=mean(mf$residuals),col="red") 

plot(mf$fitted.values,abs(mf$residuals),xlab="Fitted Values",ylab="|Residuals|") 
# ==
mc<-lm(abs(mf$residuals)~mf$fitted.values)
summary(lm(abs(mf$residuals)~mf$fitted))
## p-value=0.66>0.05 variance sabet ast

# normality - Visualization
hist(mf$residuals,col="red")
qqnorm(mf$residuals)
# ------> Normal ast

# normality - Hypothesis
ad.test(mf$residual)

plot(mf$residual[1:399],xlab="i",ylab="e(i)")  # independent 
plot(mf$residual[1:399],xlab="i",ylab="e(i)",type = "line")
acf(mf$residuals)
lines(mf$residuals, type = "c")

# Regression and Variable selection
mf <- lm(SalePrice ~. , data = df_train)
summary(mf)
mr <- lm(SalePrice~1 , data=df_train)
anova(mr, mf)
summary(mf)$coefficients

install.packages("tidyverse")
install.packages("broom")
install.packages("glmnet")
install.packages("Matrix")
library(tidyverse)
library(broom)
library(glmnet)

