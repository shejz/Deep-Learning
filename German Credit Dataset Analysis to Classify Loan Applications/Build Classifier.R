#data description:
#https://ocw.mit.edu/courses/sloan-school-of-management/15-062-data-mining-spring-2003/assignments/GermanCredit.pdf

library(DT)          # For Data Tables
library(lattice)     # The lattice add-on of Trellis graphics for R
library(knitr)       # For Dynamic Report Generation in R 
library(gplots)      # Various R Programming Tools for Plotting Data
library(ggplot2)     # An Implementation of the Grammar of Graphics 
library(ClustOfVar)  # Clustering of variables 
library(ape)         # Analyses of Phylogenetics and Evolution (as.phylo) 
library(Information) # Data Exploration with Information Theory (Weight-of-Evidence and Information Value)
library(ROCR)        # Model Performance and ROC curve
library(caret)       # Classification and Regression Training -  for any machine learning algorithms
library(rpart)       # Recursive partitioning for classification, regression and survival trees
library(rpart.utils) # Tools for parsing and manipulating rpart objects, including generating machine readable rules
library(rpart.plot)  # Plot 'rpart' Models: An Enhanced Version of 'plot.rpart'
library(randomForest)# Leo Breiman and Cutler's Random Forests for Classification and Regression 
library(party)       # A computational toolbox for recursive partitioning - Conditional inference Trees
library(bnlearn)     # Bayesian Network Structure Learning, Parameter Learning and Inference
library(DAAG)        # Data Analysis and Graphics Data and Functions
library(vcd)         # Visualizing Categorical Data
library(kernlab)     # Support Vector Machine

# Import datasets
data <- read.table("german.data")
data_num <- read.table("german.data-numeric")

# Rename Columns
colnames(data) <- c("checking_acct_status",
                    "duration_in_month",
                    "Credit_history",
                    "Purpose",
                    "Credit_amount",
                    "Savings_account/bonds",
                    "Present_employment_since",
                    "Installment_rate",
                    "Personal_status_and_sex",
                    "Others_debtors_/_guarantors",
                    "Present_residence_since",
                    "Property",
                    "Age_in_years",
                    "Other_installment_plans",
                    "Housing",
                    "Number_of_existing_credits_at_this_bank",
                    "Job",
                    "Number_of_people_being_liable_to_provide_maintenance_for",
                    "Telephone",
                    "foreign_worker",
                    "target")

# Distribution of Columns

kable(as.data.frame(colnames(data)))

table(data$checking_acct_status)
prop.table(table(data$checking_acct_status))
kable(prop.table(table(data$checking_acct_status)))

compute_dist <- function(x){
  kable(prop.table(table(x)))
}

apply(data,2,compute_dist) #2 apply on columns, -1 apply over rows

str(data)
summary(data)

# Approach 1: Fitting the data into a modeling framework
# - Compute the information value
# - Compute the weight of evidence
# - create model, test model,
# - monitor accuracy and balidate the results.

# Weight of Evidence (WOE)
func2 <- function(x,y=data$target){
  mt <- as.matrix(table(as.factor(x), as.factor(y)))
  total_abs <- mt[,1] + mt[,2]
  total_pct <- (total_abs/sum(mt)*100)
  # 1=Good, 2=Bad
  good_pct <- mt[,1]/sum(mt[,1])*100
  bad_pct <- mt[,2]/sum(mt[,2])*100
  score_grp <- ((good_pct/(good_pct+bad_pct))*10)
  WOE <- log(good_pct/bad_pct)*10                             # Weight of Evidence      
  good_bad <- ifelse(mt[,1]==mt[,2],0,1)
  IV <- ifelse(good_bad==0,0,(good_pct - bad_pct)*(WOE/10))    # Information value   
  efficiency <- abs(good_pct - bad_pct)/2
  tabledata <- as.data.frame(cbind(mt, good_pct,bad_pct, total_abs,
                                   total_pct, score_grp, WOE,IV,
                                   efficiency))
}

A1 <- func2(data$checking_acct_status)
A1

# apply to all columns
apply(data,2,func2)

# Univariate, Bivariate and Multivariate
fact_flag <- sapply(data, is.factor)

cdata <-  data[, fact_flag]
ndata <-  data[,-fact_flag]

data$target <- as.factor(data$target)


# Logistic Regression Introduction

model1 <- glm(data$target~.,data=data, family = binomial())
summary(model1)

model2 <- step(model1)
summary(model2)

# How many significant variables
sig_var <- summary(model2)$coeff[-1,4] < 0.01
names(sig_var)[sig_var==TRUE]

prob <- predict(model2, type="response")
res <- residuals(model2, type = "deviance")

# Plot
plot(predict(model2), res)


# Approach 2: Learn from the data
# - Apply ML models

as.matrix(table(as.factor(data$checking_acct_status),
                as.factor(data$target)))


# ---------------------------------------------------------------------------------------------------------------


cdata<-read.table('https://s3.amazonaws.com/hackerday.datascience/116/german.data')
# Update column Names
colnames(cdata) <- c("chk_ac_status_1",
                     "duration_month_2", "credit_history_3", "purpose_4",
                     "credit_amount_5","savings_ac_bond_6","p_employment_since_7", 
                     "instalment_pct_8", "personal_status_9","other_debtors_or_grantors_10", 
                     "present_residence_since_11","property_type_12","age_in_yrs_13",
                     "other_instalment_type_14", "housing_type_15", 
                     "number_cards_this_bank_16","job_17","no_people_liable_for_mntnance_18",
                     "telephone_19", "foreign_worker_20", 
                     "good_bad_21")

# Function 1: Create function to calculate percent distribution for factors
pct <- function(x){
  tbl <- table(x)
  tbl_pct <- cbind(tbl,round(prop.table(tbl)*100,2))
  colnames(tbl_pct) <- c('Count','Percentage')
  kable(tbl_pct)
}

pct(cdata$chk_ac_status_1)

# Function 2: Own function to calculate IV, WOE and Eefficiency 
gbpct <- function(x, y=cdata$good_bad_21){
  mt <- as.matrix(table(as.factor(x), as.factor(y))) 
  # x -> independent variable(vector), y->dependent variable(vector)
  Total <- mt[,1] + mt[,2]                          
  # Total observations
  Total_Pct <- round(Total/sum(mt)*100, 2)          
  # Total PCT
  Bad_pct <- round((mt[,1]/sum(mt[,1]))*100, 2)     
  # PCT of BAd or event or response
  Good_pct <- round((mt[,2]/sum(mt[,2]))*100, 2)   
  # PCT of Good or non-event
  Bad_Rate <- round((mt[,1]/(mt[,1]+mt[,2]))*100, 2) 
  # Bad rate or response rate
  grp_score <- round((Good_pct/(Good_pct + Bad_pct))*10, 2) 
  # score for each group
  WOE <- round(log(Good_pct/Bad_pct)*10, 2)      
  # Weight of Evidence for each group
  g_b_comp <- ifelse(mt[,1] == mt[,2], 0, 1)
  IV <- ifelse(g_b_comp == 0, 0, (Good_pct - Bad_pct)*(WOE/10)) 
  # Information value for each group
  Efficiency <- abs(Good_pct - Bad_pct)/2                       
  # Efficiency for each group
  otb<-as.data.frame(cbind(mt, Good_pct,  Bad_pct,  Total, 
                           Total_Pct,  Bad_Rate, grp_score, 
                           WOE, IV, Efficiency ))
  otb$Names <- rownames(otb)
  rownames(otb) <- NULL
  otb[,c(12,2,1,3:11)] 
  # return IV table
}

# Function 3: Normalize using Range

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}


kable(as.data.frame(colnames(cdata)))

# library(DT) # Data Table
DT::datatable(cdata[1:100,]) # First 100 observations



# Read a numeric copy: Numeric data for Neural network & Lasso
cdatanum<-read.table('https://s3.amazonaws.com/hackerday.datascience/116/german.data-numeric')
cdatanum <- as.data.frame(sapply(cdatanum, as.numeric ))

kable(as.data.frame(colnames(cdata)))

cdata$duration_month_2  <- as.numeric(cdata$duration_month_2)
cdata$credit_amount_5   <-  as.numeric(cdata$credit_amount_5 )
cdata$instalment_pct_8 <-  as.numeric(cdata$instalment_pct_8)
cdata$present_residence_since_11 <-  as.numeric(cdata$present_residence_since_11)
cdata$age_in_yrs_13        <-  as.numeric(cdata$age_in_yrs_13)
cdata$number_cards_this_bank_16    <-  as.numeric(cdata$number_cards_this_bank_16)
cdata$no_people_liable_for_mntnance_18 <-  as.numeric(cdata$no_people_liable_for_mntnance_18)

cdata$good_bad_21<-as.factor(ifelse(cdata$good_bad_21 == 1, "Good", "Bad"))
pct(cdata$good_bad_21)

op<-par(mfrow=c(1,2), new=TRUE)
plot(as.numeric(cdata$good_bad_21), ylab="Good-Bad", xlab="n", main="Good ~ Bad")
hist(as.numeric(cdata$good_bad_21), breaks=2, 
     xlab="Bad(2) and Good(1)", col="blue")

A1 <- gbpct(cdata$chk_ac_status_1)

op1<-par(mfrow=c(1,2), new=TRUE)
plot(cdata$chk_ac_status_1, cdata$good_bad_21, 
     ylab="Good-Bad", xlab="category", 
     main="Checking Account Status ~ Good-Bad ")

barplot(A1$WOE, col="brown", names.arg=c(A1$Levels), 
        main="Score:Checking Account Status",
        xlab="Category",
        ylab="WOE"
)

par(op1)

kable(A1, caption = 'Checking Account Status ~ Good-Bad')

summary(cdata$duration_month_2)

op2<-par(mfrow=c(1,2))
boxplot(cdata$duration_month_2, ylab="Loan Duration(Month)", main="Boxplot:Loan Duration")

plot(cdata$duration_month_2, cdata$good_bad_21, 
     ylab="Good-Bad", xlab="Loan Duration(Month)",
     main="Loan Duration ~ Good-Bad ")

plot(as.factor(cdata$duration_month_2), cdata$good_bad_21, 
     ylab="Good-Bad", xlab="Loan Duration(Month)",
     main="Loan Duration(Before Groupping)")

# Create some groups from contious variables
cdata$duration_month_2 <-as.factor(ifelse(cdata$duration_month_2<=6,'00-06',
                                          ifelse(cdata$duration_month_2<=12,'06-12',
                                                 ifelse(cdata$duration_month_2<=24,'12-24', 
                                                        ifelse(cdata$duration_month_2<=30,'24-30',
                                                               ifelse(cdata$duration_month_2<=36,'30-36',
                                                                      ifelse(cdata$duration_month_2<=42,'36-42','42+')))))))

plot(cdata$duration_month_2, cdata$good_bad_21,
     main="Loan Duration(after grouping) ",
     xlab="Loan Duration (Month)",
     ylab="Good-Bad")

par(op2)

A2<-gbpct(cdata$duration_month_2)

barplot(A2$WOE, col="brown", names.arg=c(A2$Levels),
        main="Loan Duration",
        xlab="Duration(Months)",
        ylab="WOE"
)

kable(A2, caption = 'Loan Duration ~ Good-Bad')

# Combine few groups together based on WOE and bad rates
cdata$credit_history_3<-as.factor(ifelse(cdata$credit_history_3 == "A30", "01.A30",
                                         ifelse(cdata$credit_history_3 == "A31","02.A31",
                                                ifelse(cdata$credit_history_3 == "A32","03.A32.A33",
                                                       ifelse(cdata$credit_history_3 == "A33","03.A32.A33",
                                                              "04.A34")))))

op3<-par(mfrow=c(1,2))
plot(cdata$credit_history_3, cdata$good_bad_21, 
     main = "Credit History ~ Good-Bad",
     xlab = "Credit History",
     ylab = "Good-Bad")

plot(cdata$credit_history_3, cdata$good_bad_21, 
     main = "Credit History(After Groupping) ~ Good-Bad ",
     xlab = "Credit History",
     ylab = "Good-Bad")

par(op3)

A3<-gbpct(cdata$credit_history_3)

barplot(A3$WOE, col="brown", names.arg=c(A3$Levels),
        main="Credit History",
        xlab="Credit History",
        ylab="WOE"
)

kable(A3, caption = "Credit History~ Good-Bad")

A4<-gbpct(cdata$purpose_4)


op4<-par(mfrow=c(1,2))
plot(cdata$purpose_4, cdata$good_bad_21, 
     main="Purpose of Loan~ Good-Bad ",
     xlab="Purpose",
     ylab="Good-Bad")

barplot(A4$WOE, col="brown", names.arg=c(A4$Levels),
        main="Purpose of Loan",
        xlab="Category",
        ylab="WOE")

par(op4)

kable(A4, caption = "Purpose of Loan~ Good-Bad")


cdata$credit_amount_5 <- as.double(cdata$credit_amount_5)
summary(cdata$credit_amount_5)

boxplot(cdata$credit_amount_5)

# Create groups based on their distribution
cdata$credit_amount_5<-as.factor(ifelse(cdata$credit_amount_5<=1400,'0-1400',
                                        ifelse(cdata$credit_amount_5<=2500,'1400-2500',
                                               ifelse(cdata$credit_amount_5<=3500,'2500-3500', 
                                                      ifelse(cdata$credit_amount_5<=4500,'3500-4500',
                                                             ifelse(cdata$credit_amount_5<=5500,'4500-5500','5500+'))))))


A5<-gbpct(cdata$credit_amount_5)





plot(cdata$credit_amount_5, cdata$good_bad_21, 
     main="Credit Ammount (After Grouping) ~ Good-Bad",
     xlab="Amount",
     ylab="Good-Bad")


barplot(A5$WOE, col="brown", names.arg=c(A5$Levels),
        main="Credit Ammount",
        xlab="Amount",
        ylab="WOE")


kable(A5, caption = "Credit Ammount ~ Good-Bad")

A6<-gbpct(cdata$savings_ac_bond_6)


plot(cdata$savings_ac_bond_6, cdata$good_bad_21, 
     main="Savings account/bonds ~ Good-Bad",
     xlab="Savings/Bonds",
     ylab="Good-Bad")


barplot(A6$WOE, col="brown", names.arg=c(A6$Levels),
        main="Savings account/bonds",
        xlab="Category",
        ylab="WOE")

kable(A6, caption = "Savings account/bonds ~ Good-Bad" )

A7<-gbpct(cdata$p_employment_since_7)

op7<-par(mfrow=c(1,2))
plot(cdata$p_employment_since_7, cdata$good_bad_21,
     main="Present employment since ~ Good-Bad",
     xlab="Employment in Years",
     ylab="Good-Bad")

barplot(A7$WOE, col="brown", names.arg=c(A7$Levels),
        main="Present employment",
        xlab="Category",
        ylab="WOE")

par(op7)

kable(A7, caption ="Present employment since ~ Good-Bad")

summary(cdata$instalment_pct_8)

op8<-par(mfrow=c(1,2))
boxplot(cdata$instalment_pct_8)
histogram(cdata$instalment_pct_8,
          main = "instalment rate in percentage of disposable income",
          xlab = "instalment percent",
          ylab = "Percent Population")
par(op8)

A8<-gbpct(cdata$instalment_pct_8)

op8_1<-par(mfrow=c(1,2))
plot(as.factor(cdata$instalment_pct_8), cdata$good_bad_21,
     main="instalment rate in percentage of disposable income ~ Good-Bad",
     xlab="Percent",
     ylab="Good-Bad")

barplot(A8$WOE, col="brown", names.arg=c(A8$Levels),
        main="instalment rate",
        xlab="Percent",
        ylab="WOE")

par(op8_1)

kable(A8, caption = "instalment rate in percentage of disposable income ~ Good-Bad")

A9<-gbpct(cdata$personal_status_9)

op9<-par(mfrow=c(1,2))
plot(cdata$personal_status_9, cdata$good_bad_21, 
     main=" Personal status",
     xlab=" Personal status",
     ylab="Good-Bad")


barplot(A9$WOE, col="brown", names.arg=c(A9$Levels),
        main="Personal status",
        xlab="Category",
        ylab="WOE")

par(op9)

kable(A9, caption =  "Personal status ~ Good-Bad")

A10<-gbpct(cdata$other_debtors_or_grantors_10)

op10<-par(mfrow=c(1,2))

plot(cdata$other_debtors_or_grantors_10, cdata$good_bad_21, 
     main="Other debtors / guarantors",
     xlab="Category",
     ylab="Good-Bad")

barplot(A10$WOE, col="brown", names.arg=c(A10$Levels),
        main="Other debtors / guarantors",
        xlab="Category",
        ylab="WOE")

par(op10)

kable(A10, caption = "Other debtors / guarantors ~ Good-Bad")

summary(cdata$present_residence_since_11)

A11<-gbpct(cdata$present_residence_since_11)

op11<-par(mfrow=c(1,2))
histogram(cdata$present_residence_since_11,
          main="Present Residence~ Good-Bad",
          xlab="Present residence Since", 
          ylab="Percent Population")

barplot(A11$WOE, col="brown", names.arg=c(A11$Levels),
        main="Present Residence",
        xlab="Category",
        ylab="WOE")
par(op11)

kable(A11, caption = "Present Residence~ Good-Bad")

A12 <- gbpct(cdata$property_type_12)

op12 <- par(mfrow = c(1,2))
plot(cdata$property_type_12, cdata$good_bad_21, 
     main = "Property Type",
     xlab="Type",
     ylab="Good-Bad")         

barplot(A12$WOE, col="brown", names.arg=c(A12$Levels),
        main="Property Type",
        xlab="Category",
        ylab="WOE")

par(op12)

kable(A12,  caption = "Property Type")

summary(cdata$age_in_yrs_13)

op13 <- par(mfrow = c(1,2))
boxplot(cdata$age_in_yrs_13)

plot(as.factor(cdata$age_in_yrs_13),  cdata$good_bad_21,
     main = "Age",
     xlab = "Age in Years",
     ylab = "Good-Bad")


par(op13)

# Group AGE - Coarse Classing (after some iterations in fine classing stage)
cdata$age_in_yrs_13 <- as.factor(ifelse(cdata$age_in_yrs_13<=25, '0-25',
                                        ifelse(cdata$age_in_yrs_13<=30, '25-30',
                                               ifelse(cdata$age_in_yrs_13<=35, '30-35', 
                                                      ifelse(cdata$age_in_yrs_13<=40, '35-40', 
                                                             ifelse(cdata$age_in_yrs_13<=45, '40-45', 
                                                                    ifelse(cdata$age_in_yrs_13<=50, '45-50',
                                                                           ifelse(cdata$age_in_yrs_13<=60, '50-60',
                                                                                  '60+'))))))))

A13<-gbpct(cdata$age_in_yrs_13)

op13_1<-par(mfrow=c(1,2))
plot(as.factor(cdata$age_in_yrs_13),  cdata$good_bad_21, 
     main="Age (After Grouping)",
     xlab="Other instalment plans",
     ylab="Good-Bad")


barplot(A13$WOE, col="brown", names.arg=c(A13$Levels),
        main="Age",
        xlab="Category",
        ylab="WOE")

par(op13_1)

kable(A13,  caption = "Age (After Grouping) ~ Good-Bad")

A14<-gbpct(cdata$other_instalment_type_14)

op14<-par(mfrow=c(1,2))

plot(cdata$other_instalment_type_14, cdata$good_bad_21, 
     main="Other instalment plans ~ Good-Bad",
     xlab="Other instalment plans",
     ylab="Good-Bad")

barplot(A14$WOE, col="brown", names.arg=c(A14$Levels),
        main="Other instalment plans",
        xlab="Category",
        ylab="WOE")

par(op14)

kable(A14, caption = "Other instalment plans ~ Good-Bad")

A15<-gbpct(cdata$housing_type_15)

op15<-par(mfrow=c(1,2))
plot(cdata$housing_type_15, cdata$good_bad_21, 
     main="Home Ownership Type",
     xlab="Type",
     ylab="Good-Bad")

barplot(A15$WOE, col="brown", names.arg=c(A15$Levels),
        main="Home Ownership Type",
        xlab="Type",
        ylab="WOE")

par(op15)

kable(A15, caption = "Home Ownership Type ~ Good-Bad")

summary(cdata$number_cards_this_bank_16)

A16<-gbpct(cdata$number_cards_this_bank_16)

op16<-par(mfrow=c(1,2))
plot(as.factor(cdata$number_cards_this_bank_16), cdata$good_bad_21,
     main="Number of credits at this bank",
     xlab="Number of Cards",
     ylab="Good-Bad")

barplot(A16$WOE, col="brown", names.arg=c(A16$Levels),
        main="Number of credits at this bank",
        xlab="Number of Cards",
        ylab="WOE")

par(op16)

kable(A16, caption = "Number of credits at this bank ~ Good-Bad")

A17<-gbpct(cdata$job_17)

op17<-par(mfrow=c(1,2))

plot(cdata$job_17, cdata$good_bad_21, 
     main="Employment Type",
     xlab="Job",
     ylab="Good-Bad")

barplot(A17$WOE, col="brown", names.arg=c(A17$Levels),
        main="Employment Type",
        xlab="Job",
        ylab="WOE")

par(op17)

kable(A17, caption = "Employment Type ~ Good-Bad")


summary(cdata$no_people_liable_for_mntnance_18)

A18<-gbpct(cdata$no_people_liable_for_mntnance_18)

op18<-par(mfrow = c(1,2))

plot(as.factor(cdata$no_people_liable_for_mntnance_18), cdata$good_bad_21, 
     main = "Number of people being liable",
     xlab = "Number of People",
     ylab = "Good-Bad")

barplot(A18$WOE, col = "brown", names.arg=c(A18$Levels),
        main = " Number of people being liable",
        xlab = "Number of People",
        ylab = "WOE")

par(op18)

kable(A18, caption = "Number of people being liable ~ Good-Bad")

A19<-gbpct(cdata$telephone_19)

op19<-par(mfrow=c(1,2))

plot(cdata$telephone_19, cdata$good_bad_21, 
     main="Telephone",
     xlab="Telephone(Yes/No)",
     ylab="Good-Bad")

barplot(A19$WOE, col="brown", names.arg=c(A19$Levels),
        main="Telephone",
        xlab="Telephone(Yeas/No)",
        ylab="WOE")

par(op19)

kable(A19, caption = "Telephone ~ Good-Bad")

A20<-gbpct(cdata$foreign_worker_20)

op20<-par(mfrow=c(1,2))

plot(cdata$foreign_worker_20, cdata$good_bad_21, 
     main="Foreign Worker",
     xlab="Foreign Worker(Yes/No)",
     ylab="Good-Bad")

barplot(A20$WOE, col="brown", names.arg=c(A20$Levels),
        main="Foreign Worker",
        xlab="Foreign Worker(Yes/No)",
        ylab="WOE")

par(op20)

kable(A20,  caption = "Foreign Worker ~ Good-Bad")

# require library(Information) 
cdata$good_bad_21<-as.numeric(ifelse(cdata$good_bad_21 == "Good", 0, 1))
IV <- Information::create_infotables(data=cdata, NULL, y="good_bad_21", 10)
IV$Summary$IV <- round(IV$Summary$IV*100,2)

IV$Tables

kable(IV$Summary)

cdata$good_bad_21<-as.factor(ifelse(cdata$good_bad_21 == 0, "Good", "Bad"))

var_list_1 <- IV$Summary[IV$Summary$IV > 2,] # 15 variables
cdata_reduced_1 <- cdata[, c(var_list_1$Variable,"good_bad_21")] #16 variables


# Step 1: Subset quantitative and qualitative variables X.quanti and X.quali

factors <- sapply(cdata_reduced_1, is.factor)
#subset Qualitative variables 
vars_quali <- cdata_reduced_1[,factors]
#vars_quali$good_bad_21<-vars_quali$good_bad_21[drop=TRUE] # remove empty factors
str(vars_quali)


#subset Quantitative variables 
vars_quanti <- cdata_reduced_1[,!factors]
str(vars_quanti)


#Step 2: Hierarchical Clustering of Variables
#requires library(ClustOfVar)
#Need help type ?hclustvar on R console

tree <- hclustvar(X.quanti=vars_quanti,X.quali=vars_quali[,-c(12)])
plot(tree, main="variable clustering")
rect.hclust(tree, k=10,  border = 1:10)

summary(tree)

# Phylogenetic trees
# require library("ape")
plot(as.phylo(tree), type = "fan",
     tip.color = hsv(runif(15, 0.65,  0.95), 1, 1, 0.7),
     edge.color = hsv(runif(10, 0.65, 0.75), 1, 1, 0.7), 
     edge.width = runif(20,  0.5, 3), use.edge.length = TRUE, col = "gray80")

summary.phylo(as.phylo(tree))

part<-cutreevar(tree,10)
print(part)

summary(part)

keep<- c(1:8,12,13,21)
cdata_reduced_2 <- cdata[,keep]
str(cdata_reduced_2)

div_part <- sort(sample(nrow(cdata_reduced_2), 
                        nrow(cdata_reduced_2)*.6))

#select training sample 
train<-cdata_reduced_2[div_part,] # 70% here
pct(train$good_bad_21)

# put remaining into test sample
test<-cdata_reduced_2[-div_part,] # rest of the 30% data goes here
pct(test$good_bad_21)

pct(cdata_reduced_2$good_bad_21)

div_part_1 <- createDataPartition(y = cdata_reduced_2$good_bad_21, p = 0.7, list = F)

# Training Sample
train_1 <- cdata_reduced_2[div_part_1,] # 70% here
pct(train_1$good_bad_21)


# Test Sample
test_1 <- cdata_reduced_2[-div_part_1,] # rest of the 30% data goes here
pct(test_1$good_bad_21)

save(train_1, file="train_1.RData")
save(test_1, file="test_1.RData")

# For neural network we would need contious data
# Sampling for Neural Network - It can be used for other modeling as well
div_part_2 <- createDataPartition(y = cdatanum[,25], p = 0.7, list = F)

# Training Sample for Neural Network
train_num <- cdatanum[div_part_2,] # 70% here

# Test Sample for Neural Network
test_num <- cdatanum[-div_part_2,] # rest of the 30% data goes here

# Save for the future
save(train_num, file="train_num.RData")
save(test_num, file="test_num.RData")

# Model: Stepwise Logistic Regression Model
m1 <- glm(good_bad_21~.,data=train_1,family=binomial())
summary(m1)
m1 <- step(m1)

# List of significant variables and features with p-value <0.01
significant.variables <- summary(m1)$coeff[-1,4] < 0.01
names(significant.variables)[significant.variables == TRUE]

prob <- predict(m1, type = "response")
res <- residuals(m1, type = "deviance")

#Plot Residuals
plot(predict(m1), res,
     xlab="Fitted values", ylab = "Residuals",
     ylim = max(abs(res)) * c(-1,1))

## CIs using profiled log-likelihood
confint(m1)

## CIs using standard errors
confint.default(m1)

## odds ratios and 95% CI
exp(cbind(OR = coef(m1), confint(m1)))

#score test data set
test_1$m1_score <- predict(m1,type='response',test_1)
m1_pred <- prediction(test_1$m1_score, test_1$good_bad_21)
m1_perf <- performance(m1_pred,"tpr","fpr")

#ROC
plot(m1_perf, lwd=2, colorize=TRUE, main="ROC m1: Logistic Regression Performance")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

# Plot precision/recall curve
m1_perf_precision <- performance(m1_pred, measure = "prec", x.measure = "rec")
plot(m1_perf_precision, main="m1 Logistic:Precision/recall curve")

# Plot accuracy as function of threshold
m1_perf_acc <- performance(m1_pred, measure = "acc")
plot(m1_perf_acc, main="m1 Logistic:Accuracy as function of threshold")


#KS, Gini & AUC m1
m1_KS <- round(max(attr(m1_perf,'y.values')[[1]]-attr(m1_perf,'x.values')[[1]])*100, 2)
m1_AUROC <- round(performance(m1_pred, measure = "auc")@y.values[[1]]*100, 2)
m1_Gini <- (2*m1_AUROC - 100)
cat("AUROC: ",m1_AUROC,"\tKS: ", m1_KS, "\tGini:", m1_Gini, "\n")

#Model
m1_1 <- glm(good_bad_21~chk_ac_status_1+duration_month_2
            +savings_ac_bond_6+instalment_pct_8,
            data=train_1,family=binomial())
step(m1_1)

# Model Scoring
test_1$m1_1_score <- predict(m1_1,type='response',test_1)
m1_1_pred <- prediction(test_1$m1_1_score,test_1$good_bad_21)
m1_1_perf <- performance(m1_1_pred,"tpr","fpr")

#Model Performance plot
plot(m1_1_perf, lwd=2, colorize=TRUE,main = " ROC m1_1: Logistic Regression with selected variables")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

# Plot precision/recall curve
m1_1_perf_precision <- performance(m1_1_pred, measure = "prec", x.measure = "rec")
plot(m1_1_perf_precision, main="m1_1 Logistic:Precision/recall curve")

# Plot accuracy as function of threshold
m1_1_perf_acc <- performance(m1_1_pred, measure = "acc")
plot(m1_1_perf_acc, main="m1_1 Logistic:Accuracy as function of threshold")


#KS & AUC m1_1
m1_1_AUROC <- round(performance(m1_1_pred, measure = "auc")@y.values[[1]]*100, 2)
m1_1_KS <- round(max(attr(m1_1_perf,'y.values')[[1]]-attr(m1_1_perf,'x.values')[[1]])*100, 2)
m1_1_Gini <- (2*m1_1_AUROC - 100)
cat("AUROC: ",m1_1_AUROC,"\tKS: ", m1_1_KS, "\tGini:", m1_1_Gini, "\n")

# Requires library(rpart)
m2 <- rpart(good_bad_21~.,data=train_1)
# Print tree detail
printcp(m2)

# Tree plot
plot(m2, main="Tree:Recursive Partitioning");text(m2);

# Better version of plot
prp(m2,type=2,extra=1,  main="Tree:Recursive Partitioning")

# score test data
test_1$m2_score <- predict(m2,type='prob',test_1)
m2_pred <- prediction(test_1$m2_score[,2],test_1$good_bad_21)
m2_perf <- performance(m2_pred,"tpr","fpr")

# MOdel performance plot
plot(m2_perf, lwd=2, colorize=TRUE, main="ROC m2: Traditional Recursive Partitioning")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

# Plot precision/recall curve
m2_perf_precision <- performance(m2_pred, measure = "prec", x.measure = "rec")
plot(m2_perf_precision, main="m2 Recursive Partitioning:Precision/recall curve")

# Plot accuracy as function of threshold
m2_perf_acc <- performance(m2_pred, measure = "acc")
plot(m2_perf_acc, main="m2 Recursive Partitioning:Accuracy as function of threshold")

# KS & AUC m1
m2_AUROC <- round(performance(m2_pred, measure = "auc")@y.values[[1]]*100, 2)
m2_KS <- round(max(attr(m2_perf,'y.values')[[1]]-attr(m2_perf,'x.values')[[1]])*100, 2)
m2_Gini <- (2*m2_AUROC - 100)
cat("AUROC: ",m2_AUROC,"\tKS: ", m2_KS, "\tGini:", m2_Gini, "\n")


# Requires library(rpart)
# Fit Model 
m2_1 <- rpart(good_bad_21~.,data=train_1,parms=list(prior=c(.9,.1)),cp=.0002) #- build model using 90%-10% priors
m2_1 <- rpart(good_bad_21~.,data=train_1,parms=list(prior=c(.8,.2)),cp=.0002) #- build model using 80%-20% priors
m2_1 <- rpart(good_bad_21~.,data=train_1,parms=list(prior=c(.7,.3)),cp=.0002)  #- build model using 70%-30% priors
m2_1 <- rpart(good_bad_21~.,data=train_1,parms=list(prior=c(.75,.25)),cp=.0002) #- build model using 75%-25% priors
m2_1 <- rpart(good_bad_21~.,data=train_1,parms=list(prior=c(.6,.4)),cp=.0002) #- build model using 60%-40% priors

# Print tree detail
printcp(m2_1)

# plot trees
plot(m2_1, main="m2_1-Recursive Partitioning - Using Bayesian N 70%-30%");text(m2_1);

prp(m2_1,type=2,extra=1, main="m2_1-Recursive Partitioning - Using Bayesian N 70%-30%")

test_1$m2_1_score <- predict(m2_1,type='prob',test_1)
m2_1_pred <- prediction(test_1$m2_1_score[,2],test_1$good_bad_21)
m2_1_perf <- performance(m2_1_pred,"tpr","fpr")

# MOdel performance plot
plot(m2_1_perf, colorize=TRUE, main="ROC m2_1: Recursive Partitioning - Using Bayesian N 70%-30%")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

# Plot precision/recall curve
m2_1_perf_precision <- performance(m2_1_pred, measure = "prec", x.measure = "rec")
plot(m2_1_perf_precision, main="m2_1 Recursive Partitioning:Precision/recall curve")

# Plot accuracy as function of threshold
m2_1_perf_acc <- performance(m2_1_pred, measure = "acc")
plot(m2_1_perf_acc, main="m2_1 Recursive Partitioning:Accuracy as function of threshold")

# KS & AUC m2_1
m2_1_AUROC <- round(performance(m2_1_pred, measure = "auc")@y.values[[1]]*100, 2)
m2_1_KS <- round(max(attr(m2_1_perf,'y.values')[[1]]-attr(m2_1_perf,'x.values')[[1]])*100, 2)
m2_1_Gini <- (2*m2_1_AUROC - 100)
cat("AUROC: ",m2_1_AUROC,"\tKS: ", m2_1_KS, "\tGini:", m2_1_Gini, "\n")

#prints complexity and out of sample error
printcp(m2)

#plots complexity vs. error
plotcp(m2)

#prints complexity and out of sample error
printcp(m2_1)

#plots complexity vs. error
plotcp(m2_1)

# ROC Comparision
plot(m2_perf,  colorize=TRUE, lwd=2,lty=1, main='Recursive partitioning:Model Performance Comparision (m2 ROC)') 
plot(m2_1_perf, col='blue',lty=2, add=TRUE); # simple tree
legend(0.4,0.4,
       c("m2: Traditional", "m2_1: Bayesian -70%-30%"),
       col=c('orange', 'blue'),
       lwd=3);
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

#print rules for all classes
#rpart.lists(m2)
#rpart.rules(m2)
#rpart.lists(m2_1)
#rpart.rules.table(m2_1)

# Requires library(randomForest)
m3 <- randomForest(good_bad_21 ~ ., data = train_1)

m3_fitForest <- predict(m3, newdata = test_1, type="prob")[,2]
m3_pred <- prediction( m3_fitForest, test_1$good_bad_21)
m3_perf <- performance(m3_pred, "tpr", "fpr")

#plot variable importance
varImpPlot(m3, main="Random Forest: Variable Importance")

# Model Performance plot
plot(m3_perf,colorize=TRUE, lwd=2, main = "m3 ROC: Random Forest", col = "blue")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

# Plot precision/recall curve
m3_perf_precision <- performance(m3_pred, measure = "prec", x.measure = "rec")
plot(m3_perf_precision, main="m3 Random Forests:Precision/recall curve")

# Plot accuracy as function of threshold
m3_perf_acc <- performance(m3_pred, measure = "acc")
plot(m3_perf_acc, main="m3 Random Forests:Accuracy as function of threshold")

#KS & AUC  m3
m3_AUROC <- round(performance(m3_pred, measure = "auc")@y.values[[1]]*100, 2)
m3_KS <- round(max(attr(m3_perf,'y.values')[[1]] - attr(m3_perf,'x.values')[[1]])*100, 2)
m3_Gini <- (2*m3_AUROC - 100)
cat("AUROC: ",m3_AUROC,"\tKS: ", m3_KS, "\tGini:", m3_Gini, "\n")

# requires library(party)
set.seed(123456742)
m3_1 <- cforest(good_bad_21~., control = cforest_unbiased(mtry = 2, ntree = 50), data = train_1)
#plot(m3_1)

# Variable Importance
kable(as.data.frame(varimp(m3_1)))

# Model Summary
summary(m3_1)

# Model Performance
m3_1_fitForest <- predict(m3, newdata = test_1, type = "prob")[,2]
m3_1_pred <- prediction(m3_1_fitForest, test_1$good_bad_21)
m3_1_perf <- performance(m3_1_pred, "tpr", "fpr")

# Model Performance Plot
plot(m3_1_perf, colorize=TRUE, lwd=2, main = " m3_1 ROC: Conditional Random Forests")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

# Plot precision/recall curve
m3_1_perf_precision <- performance(m3_1_pred, measure = "prec", x.measure = "rec")
plot(m3_1_perf_precision, main="m3_1 Conditional Random Forests:Precision/recall curve")

# Plot accuracy as function of threshold
m3_1_perf_acc <- performance(m3_1_pred, measure = "acc")
plot(m3_1_perf_acc, main="m3_1 Conditional Random Forests:Accuracy as function of threshold")

# KS & AUC m3_1
m3_1_AUROC <- round(performance(m3_1_pred, measure = "auc")@y.values[[1]]*100, 2)
m3_1_KS <- round(max(attr(m3_perf,'y.values')[[1]] - attr(m3_perf,'x.values')[[1]])*100, 2)
m3_1_Gini <- (2*m3_1_AUROC - 100)
cat("AUROC: ",m3_1_AUROC,"\tKS: ", m3_1_KS, "\tGini:", m3_1_Gini, "\n")

#logistic regression model using important variables from RF model
m3_2 <- glm(good_bad_21~.+credit_history_3:p_employment_since_7
            + credit_history_3:age_in_yrs_13
            + chk_ac_status_1:p_employment_since_7
            + chk_ac_status_1:savings_ac_bond_6
            + duration_month_2:purpose_4, data=train_1,family=binomial())


m3_2 <- step(m3_2)

# Model Performance
test_1$m3_2_score <- predict(m3_2,type='response',test_1)
m3_2_pred <- prediction(test_1$m3_2_score,test_1$good_bad_21)
m3_2_perf <- performance(m3_2_pred,"tpr","fpr")

# ROC Plot
plot(m3_2_perf, colorize=TRUE, lwd=2, main="ROC m3_2:Improve Logistic Results using Random Forest")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

# Plot precision/recall curve
m3_2_perf_precision <- performance(m3_2_pred, measure = "prec", x.measure = "rec")
plot(m3_2_perf_precision, main="m3_2 Improve Logistic:Precision/recall curve")

# Plot accuracy as function of threshold
m3_2_perf_acc <- performance(m3_2_pred, measure = "acc")
plot(m3_2_perf_acc, main="m3_2 Improve Logistic:Accuracy as function of threshold")

#KS & AUC m3_2
m3_2_AUROC <- round(performance(m3_2_pred, measure = "auc")@y.values[[1]]*100, 2)
m3_2_KS <- round(max(attr(m3_2_perf,'y.values')[[1]]-attr(m3_2_perf,'x.values')[[1]])*100, 2)
m3_2_Gini <- (2*m3_2_AUROC - 100)
cat("AUROC: ",m3_2_AUROC,"\tKS: ", m3_2_KS, "\tGini:", m3_2_Gini, "\n")


# Conditional Tree
#library(party)
m4 <- ctree(good_bad_21~.,data=train_1)
plot(m4, main="m4: Conditional inference Tree",col="blue");

resultdfr <- as.data.frame(do.call("rbind", treeresponse(m4, newdata = test_1)))
test_1$m4_score <- resultdfr[,2]
m4_pred <- prediction(test_1$m4_score,test_1$good_bad_21)
m4_perf <- performance(m4_pred,"tpr","fpr")

# Model Performance
plot(m4_perf, colorize=TRUE, lwd=2, main="ROC m4: Conditional inference Tree")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

# Plot precision/recall curve
m4_perf_precision <- performance(m4_pred, measure = "prec", x.measure = "rec")
plot(m4_perf_precision, main="m4 CIT:Plot precision/recall curve")

# Plot accuracy as function of threshold
m4_perf_acc <- performance(m4_pred, measure = "acc")
plot(m4_perf_acc, main="m4 CIT:Plot accuracy as function of threshold")

#KS & AUC m4
m4_AUROC <- round(performance(m4_pred, measure = "auc")@y.values[[1]]*100, 2)
m4_KS <- round(max(attr(m4_perf,'y.values')[[1]]-attr(m4_perf,'x.values')[[1]])*100, 2)

m4_Gini <- (2*m4_AUROC - 100)
cat("AUROC: ",m4_AUROC,"\tKS: ", m4_KS, "\tGini:", m4_Gini, "\n")

# Bayesian Learn Model
#library(bnlearn)
train_2<-train_1
train_2$duration_month_2 <- as.factor(train_2$duration_month_2)
train_2$credit_amount_5 <- as.factor(train_2$credit_amount_5)
train_2$instalment_pct_8 <- as.factor(train_2$instalment_pct_8)
train_2$age_in_yrs_13 <- as.factor(train_2$age_in_yrs_13)

bn.gs <- gs(train_2)
bn.gs

bn2 <- iamb(train_2)
bn2

bn3 <- fast.iamb(train_2)
bn3

bn4 <- inter.iamb(train_2)
bn4

compare(bn.gs, bn2)

compare(bn.gs, bn3)

compare(bn.gs, bn4)

bn.hc <- hc(train_2, score = "aic")
bn.hc

compare(bn.hc, bn.gs)

opm5<-par(mfrow = c(1,2))
plot(bn.gs, main = "Constraint-based algorithms")
plot(bn.hc, main = "Hill-Climbing")

par(opm5)
modelstring(bn.hc)

res2 = hc(train_2)
fitted2 = bn.fit(res2, train_2)
fitted2

# KSVM - Kernel Support Vector Machines
m7_1 <- ksvm(good_bad_21 ~ ., 
             data = train_1, 
             kernel = "vanilladot")

m7_1_pred <- predict(m7_1, test_1[,1:10], type="response")
head(m7_1_pred)

# Model accuracy:
table(m7_1_pred, test_1$good_bad_21)

#agreement
m7_1_accuracy  <- (m7_1_pred == test_1$good_bad_21)
pct(m7_1_accuracy)

# Compute at the prediction scores
m7_1_score <- predict(m7_1,test_1, type="decision")
m7_1_pred <- prediction(m7_1_score, test_1$good_bad_21)

# Plot ROC curve
m7_1_perf <- performance(m7_1_pred, measure = "tpr", x.measure = "fpr")
plot(m7_1_perf, colorize=TRUE, lwd=2, main="m7_1 SVM:Plot ROC curve - Vanilladot")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

# Plot precision/recall curve
m7_1_perf_precision <- performance(m7_1_pred, measure = "prec", x.measure = "rec")
plot(m7_1_perf_precision, main="m7_1 SVM:Plot precision/recall curve")

# Plot accuracy as function of threshold
m7_1_perf_acc <- performance(m7_1_pred, measure = "acc")
plot(m7_1_perf_acc, main="m7_1 SVM:Plot accuracy as function of threshold")

# Model Performance

#KS & AUC m7_1
m7_1_AUROC <- round(performance(m7_1_pred, measure = "auc")@y.values[[1]]*100, 2)
m7_1_KS <- round(max(attr(m7_1_perf,'y.values')[[1]]-attr(m7_1_perf,'x.values')[[1]])*100, 2)
m7_1_Gini <- (2*m7_1_AUROC - 100)
cat("AUROC: ",m7_1_AUROC,"\tKS: ", m7_1_KS, "\tGini:", m7_1_Gini, "\n")

#Requires library(kernlab)
# Model Improvement with  Gaussian RBF kernel
m7_2 <- ksvm(good_bad_21 ~ ., data = train_1, kernel = "rbfdot")
m7_2_pred <- predict(m7_2, test_1[,1:10], type="response")
head(m7_2_pred)

# Model accuracy:
table(m7_2_pred, test_1$good_bad_21)

#agreement
m7_2_accuracy  <- (m7_2_pred == test_1$good_bad_21)
pct(m7_2_accuracy)

# Compute at the prediction scores
m7_2_score <- predict(m7_2,test_1, type="decision")
m7_2_pred <- prediction(m7_2_score, test_1$good_bad_21)


# Plot ROC curve
m7_2_perf <- performance(m7_2_pred, measure = "tpr", x.measure = "fpr")
plot(m7_2_perf, colorize=TRUE, lwd=2, main="SVM:Plot ROC curve - RBF", col="blue")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

# Plot precision/recall curve
m7_2_perf_precision <- performance(m7_2_pred, measure = "prec", x.measure = "rec")
plot(m7_2_perf_precision, main="m7_2 SVM:Plot precision/recall curve")

# Plot accuracy as function of threshold
m7_2_perf_acc <- performance(m7_2_pred, measure = "acc")
plot(m7_2_perf_acc, main="m7_2 SVM:Plot accuracy as function of threshold")

# Model Performance
#KS &AUC m7_2
m7_2_AUROC <- round(performance(m7_2_pred, measure = "auc")@y.values[[1]]*100, 2)
m7_2_KS <- round(max(attr(m7_2_perf,'y.values')[[1]]-attr(m7_2_perf,'x.values')[[1]])*100, 2)
m7_2_Gini <- (2*m7_2_AUROC - 100)
cat("AUROC: ",m7_2_AUROC,"\tKS: ", m7_2_KS, "\tGini:", m7_2_Gini, "\n")

# ROC Comparision
plot(m7_1_perf, col='blue', lty=1, main='SVM:Model Performance Comparision (m7 ROC)') 
plot(m7_2_perf, col='green',lty=2, add=TRUE); # simple tree
legend(0.5,0.4,
       c("m7_1: SVM vanilladot", "m7_2: SVM RBF kernel"),
       col=c('blue', 'green'),
       lwd=3);
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);# random line

# Precision Comparision
plot(m7_1_perf_precision, col='blue', lty=1, main='SVM:Model Performance Comparision (m7 precision/recall)') 
plot(m7_2_perf_precision, col='green',lty=2, add=TRUE); # simple tree
legend(0.2,0.85,c("m7_1: SVM vanilladot", "m7_2: SVM RBF kernel"),
       col=c('blue', 'green'),lwd=3);

# Plot accuracy as function of threshold
plot(m7_1_perf_acc, col='blue', lty=1, main='SVM:Model accuracy as function of threshold (m7)') 
plot(m7_2_perf_acc, col='green',lty=2, add=TRUE); # simple tree
legend(-1,0.5,c("m7_1: SVM vanilladot", "m7_2: SVM RBF kernel"),
       col=c('blue', 'green'),lwd=3);

#Neural Network
library(nnet)
library(NeuralNetTools)
library(e1071)

# Normalize
train_num$V25 <- as.factor(train_num$V25)
test_num$V25 <- as.factor(test_num$V25)

train_num_norm <- as.data.frame(lapply(train_num[,1:24], normalize ))
test_num_norm <- as.data.frame(lapply(test_num[,1:24], normalize ))

train_num_norm$V25 <- as.factor(ifelse(train_num$V25 == 1, 1, 0))
test_num_norm$V25 <- as.factor(ifelse(test_num$V25 == 1, 1, 0))

# train_num_norm <- as.data.frame(lapply(train_num[,1:24], scale )) # use scale if normal
# test_num_norm <- as.data.frame(lapply(test_num[,1:24], scale ))   # use scale if normal


# build the neural network (NN) formula
a <- colnames(train_num[,1:24])
mformula <- as.formula(paste('V25 ~ ' , paste(a,collapse='+')))

set.seed(1234567890)
train_nn <- train_num_norm
test_nn <- test_num_norm

# Modeling
m8 <- nnet(V25~., data=train_nn,size=20,maxit=10000,
           decay=.001, linout=F, trace = F)

table(test_nn$V25,predict(m8,newdata=test_nn, type="class"))

m8_pred <- prediction(predict(m8, newdata=test_nn, type="raw"),test_nn$V25)
m8_perf <- performance(m8_pred,"tpr","fpr")

# Model Performance
#ROC
plot(m8_perf,lwd=2, colerize=TRUE, main="m8:ROC - Neural Network")
abline(a=0,b=1)

# Residula plots
plot(m8$residuals)

#KS, Gini & AUC m8
m8_AUROC <- round(performance(m8_pred, measure = "auc")@y.values[[1]]*100, 2)
m8_KS <- round(max(attr(m8_perf,'y.values')[[1]] - attr(m8_perf,'x.values')[[1]])*100, 2)
m8_Gini <- (2*m8_AUROC - 100)
cat("AUROC: ",m8_AUROC,"\tKS: ", m8_KS, "\tGini:", m8_Gini, "\n")

# get the weights and structure in the right format
wts <- neuralweights(m8)
struct <- wts$struct
wts <- unlist(wts$wts)
plotnet(m8, struct=struct)

mat_in <- train_nn[,1:24]
grps <- apply(mat_in, 2, quantile, seq(0, 1, by = 0.2))
pred_val <- as.data.frame(pred_sens(mat_in, m8, 'V1', 100, grps, 'V25'))
head(pred_val, 10)

#Compare ROC Performance of Models
plot(m1_perf, col='blue', lty=1, main='ROCs: Model Performance Comparision') # logistic regression
plot(m2_perf, col='gold',lty=2, add=TRUE); # simple tree
plot(m2_1_perf, col='dark orange',lty=3, add=TRUE); #tree with 90/10 prior
plot(m3_perf, col='green',add=TRUE,lty=4); # random forest
plot(m4_perf, col='dark gray',add=TRUE,lty=5); # Conditional Inference Tree
plot(m3_2_perf, col='dark green',add=TRUE,lty=6); # Improved logistic regression using random forest
plot(m7_2_perf, col='black',add=TRUE,lty=7); # Support Vector Machine (SVM)
plot(m8_perf, col='red',add=TRUE,lty=8); # Neural Network
plot(m9_2_perf, col='brown',add=TRUE,lty=9); # Neural Network
legend(0.6,0.5,
       c('m1:logistic reg','m2:Recursive Partitioning','m2_1:Recursive Partitioning - Bayesian', 
         'm3:random forest', "m4:condtn inference tree", "m3_2:Improved Logistic", "m7_2:SVM", 
         "m8:Neural Net", "m9:Lasso"),
       col=c('blue','gold', 'orange','green', 'dark gray', 'dark green', "black", "red","brown"),
       lwd=3);
lines(c(0,1),c(0,1),col = "gray", lty = 4 ) # random line

# Performance Table
models <- c('m1:Logistic regression', 'm1:Logistic regression - selected vars',
            'm2:Recursive partitioning','m2_1:Recursive partitioning-Bayesian 70/30% prior', 
            'm3:Random forest', "m3_2: Improved Logistic",
            "m4:Conditional inference tree",  
            "m7_1:SVM (vanilla)", "m7_2:SVM (RBF)", "m8:Neural net")

# AUCs
models_AUC <- c(m1_AUROC, m1_1_AUROC, m2_AUROC, m2_1_AUROC, 
                m3_AUROC,m3_2_AUROC, m4_AUROC,
                m7_1_AUROC, m7_2_AUROC, m8_AUROC)
# KS
models_KS <- c(m1_KS, m1_1_KS, m2_KS, m2_1_KS, 
               m3_KS,m3_2_KS,m4_KS, 
               m7_1_KS, m7_2_KS, m8_KS)

# Gini
models_Gini <- c(m1_Gini, m1_1_Gini, m2_Gini, m2_1_Gini, 
                 m3_Gini, m3_2_Gini, m4_Gini,
                 m7_1_Gini, m7_2_Gini, m8_Gini )

# Combine AUC and KS
#model_performance_metric <- as.data.frame(cbind(models, models_AUC, models_KS, models_Gini))
model_performance_metric <- (cbind(models, models_AUC, models_KS, models_Gini))

# Colnames 
colnames(model_performance_metric) <- c("Model", "AUC", "KS", "Gini")

# Display Performance Reports
kable(model_performance_metric, caption ="Comparision of Model Performances")

