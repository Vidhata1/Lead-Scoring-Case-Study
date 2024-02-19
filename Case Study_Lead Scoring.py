#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv("Leads.csv")
df.head() ## first 5 rows details


# In[5]:


df.tail() ## last 5 rows details


# In[6]:


df.shape ## checking how many rows and columns


# In[7]:


df.describe() ## checking the statistical values( mean, mode,25%,50%,75%..etc)


# In[8]:


df.info() ## Checking the column types of the data given


# In[9]:


df.isnull().sum() ## checking for null values


# Clean the data

# There are so many null values so that
# . we can drop some columns which have null values(we can drop the columns which have more than 40% null values)
# . we can drop some columns irrelevant columns
# . we can impute values for some colums with mean or median based on tha type of variable
# . we can drop some variables won't be of any use in our analysis (example city,country, etc)
# . Few columns have 'Select' in their entries, it's basically the value when one doesn't choose any option.So, it can be replace with null values.

# separating the given data into numerical and categorical variables

# why we are doing this because we can identify easily which variables are numerical and which are categorical variables

# 

# In[11]:


numeric_data = df.select_dtypes(include=[np.number])
categorical_data = df.select_dtypes(exclude=[np.number])

numeric_data
categorical_data


# With numeric variables, we can impute missing values using mean, mode or median, replace invalid values.
# With categorical variables, we can impute missing values with median,mode new category ,or frequently occurring category.

# In[12]:


# Get the value counts of all the columns

for column in df:
    print(df[column].astype('category').value_counts())
    print('___________________________________________________')


# In[13]:


#Removing column Prospect ID
df.drop(labels=['Prospect ID','Lead Number'],axis=1,inplace=True)
df.head()


# In[14]:


# Replacing Select with NaN
df.replace('Select',np.NaN,inplace=True)
df.head()


# In[15]:


#Checking the  percentage of Null Values in each column
round(100*(df.isna().sum()/len(df)),2)


# In[16]:


#Dropping the columns which has more than 40% of null values
df.drop(['How did you hear about X Education','Lead Quality','Lead Profile','Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score','Asymmetrique Profile Score'],axis=1,inplace=True)


# In[17]:


#Dropping the irrelevant columns
df.drop(['A free copy of Mastering The Interview','I agree to pay the amount through cheque','Tags','Last Notable Activity','Last Activity'],axis=1,inplace=True)


# In[18]:


# Again Checking the  percentage of Null Values in each column
round(100*(df.isna().sum()/len(df)),2)


# In[19]:


#Checking the distribution for the column TotalVisits
sns.distplot(df.TotalVisits)


# In[20]:


#imputing it with the median
df.TotalVisits.fillna(df.TotalVisits.median(),inplace=True)


# In[21]:


#Checking the distribution of the column Page Views Per Visit
sns.distplot(df['Page Views Per Visit'])


# In[22]:


df['Page Views Per Visit'].fillna(df['Page Views Per Visit'].median(),inplace=True)


# The above graph shows a skewed graph so that we are imputing values with median

# In[23]:


#Rechecking the null values
round(df.isna().sum()/len(df),2)


# In[24]:


# droping city and country variables won't be any use in our analysis
df.drop(['City','Country'],axis=1,inplace=True)


# In[25]:


df['What matters most to you in choosing a course'].value_counts()


# The variable What matters most to you in choosing a course has the level Better Career Prospects 6528 times while the other two levels appear once twice and once respectively. So we should drop this column as well.

# In[26]:


df.drop(['What matters most to you in choosing a course'], axis = 1, inplace=True)


# In[27]:


# Checking the value counts for the Specialization variable
df['Specialization'].value_counts()


# In[28]:


#Checking the values of Specialization column in percentage
100*df.Specialization.value_counts(normalize=True,dropna=False)


# In[29]:


#imputing the null values in the Specialization column with the mode
df.Specialization.fillna(df.Specialization.mode()[0],inplace=True)


# In[30]:


df['What is your current occupation'].value_counts


# In[31]:


#imputing the null values in the 'What is your current occupation' column with the mode
df['What is your current occupation'].fillna(df['What is your current occupation'].mode()[0],inplace=True)


# In[32]:


# Check the number of null values again
df.isnull().sum()


# In[33]:


df['Lead Source'].value_counts


# For this null values are only 0.3 so we are removeing those rows

# In[34]:


# Drop the null values rows in the column 'Lead Source'
df = df[~pd.isnull(df['Lead Source'])]


# In[35]:


# Recheck the number of null values 
df.isnull().sum()


# In[36]:


#Rechecking the null values
round(df.isna().sum()/len(df),2)


# Data is cleaned,so there are no null values

# # Prepare the data for Model Building

# In[37]:


#separating the data into numerical and categorical variables
numeric_data = df.select_dtypes(include=[np.number])
categorical_data = df.select_dtypes(exclude=[np.number])

numeric_data
categorical_data


# Numerical Analysis

# In[38]:


#Checking correlations of numeric values
# figure size
plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[39]:


#Check the % of Data that has Converted Values = 1:
Converted = (sum(df['Converted'])/len(df['Converted'].index))*100
Converted


# # Converted is the target variable, Indicates whether a lead has been successfully converted (1) or not (0).

# Univariate Analysis

# In[40]:


s1=sns.countplot(x = "Lead Origin", hue = "Converted", data = df)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# Inference:

# 1.API and Landing Page Submission have 30-35% conversion rate but count of lead originated from them are considerable.
# 2.Lead Add Form has more than 90% conversion rate but count of lead are not very high.
# 3.Lead Import are very less in count.

# Note: To improve overall lead conversion rate, we need to focus more on improving lead converion of API and Landing Page Submission origin and generate more leads from Lead Add Form.

# In[41]:


fig, axs = plt.subplots(figsize = (15,7.5))
s2=sns.countplot(x = "Lead Source", hue = "Converted", data = df)
s2.set_xticklabels(s2.get_xticklabels(),rotation=90)
plt.show()


# Inference:

# 1.Google and Direct traffic generates maximum number of leads.
# 2.Conversion Rate of reference leads and leads through welingak website is high.

# Note: To improve overall lead conversion rate, focus should be on improving lead converion of olark chat, organic search, direct traffic, and google leads and generate more leads from reference and welingak website.

# Bivariate Analysis

# In[42]:


fig, axs = plt.subplots(1,2,figsize = (15,7.5))
sns.countplot(x = "Do Not Email", hue = "Converted", data = df, ax = axs[0])
sns.countplot(x = "Do Not Call", hue = "Converted", data = df, ax = axs[1])


# Inference:

# 1.As we can see so many customers choose the option for Do Not Emails and Do Not Call

# In[43]:


# Visualizing the correlation between all set of usable columns
plt.figure(figsize=(5, 5))
sns.heatmap(df.corr(), cmap="YlGnBu",annot=True)


# Inference:

# 1.There is positive correlation between Total Time Spent on Website and Conversion
# 2.There is almost no correlation in Page Views Per Visit and TotalVisits with Conversion

# In[44]:


# Visualizing the correlation between all set of columns
plt.figure(figsize=(24, 16))
sns.heatmap(df.corr(), cmap="YlGnBu",annot=True)


# Outlier treatment

# In[45]:


df['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[46]:


sns.boxplot(df['TotalVisits'])


# There is an outliers in the TotalVisits so we can treat them as Remove top & bottom 1% of the Column Outlier values

# In[47]:


Q3 = df.TotalVisits.quantile(0.99)
leads =df[(df.TotalVisits <= Q3)]
Q1 = df.TotalVisits.quantile(0.01)
leads =df[(df.TotalVisits >= Q1)]
sns.boxplot(y=df['TotalVisits'])
plt.show()


# The outliers are cleared for TotalVisits

# In[48]:


sns.boxplot(y = 'TotalVisits', x = 'Converted', data = df)


# Inference:

# 1.Median for converted and not converted leads are the close.
# 2.Nothng conclusive can be said on the basis of Total Visits

# In[49]:


df['Total Time Spent on Website'].describe()


# In[50]:


sns.boxplot(df['Total Time Spent on Website'])


# There are no major Outliers for the Total Time Spent on Website variable we don't do any Outlier Treatment for this above Column
# 
# 

# In[51]:


sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = df)


# Inference:

# 1.Leads spending more time on the weblise are more likely to be converted.
# 2.Website should be made more engaging to make leads spend more time.

# In[52]:


df['Page Views Per Visit'].describe()


# In[53]:


sns.boxplot(df['Page Views Per Visit'])


# As we can see there are a number of outliers in the data so We will cap the outliers to 95% value for analysis.

# In[54]:


percentiles = df['Page Views Per Visit'].quantile([0.05,0.95]).values
df['Page Views Per Visit'][df['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
df['Page Views Per Visit'][df['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]


# In[55]:


sns.boxplot(df['Page Views Per Visit'])


# In[56]:


sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data = df)


# Inference:

# 1.Median for converted and unconverted leads is the same.
# 2.Nothing can be said specifically for lead conversion from Page Views Per Visit

# In[57]:


#checking missing values in leftover columns/
round(100*(df.isnull().sum()/len(df.index)),2)


# There are no missing values in the other columns to be analyzed further

# Converting Binary variables (Yes/No) to 1/0

# In[58]:


# List of variables to map

varlist =  ['Do Not Email', 'Do Not Call']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
df[varlist] = df[varlist].apply(binary_map)


# Dummy Variable Creation

# In[59]:


#Again getting categorical columns
cat_cols= df.select_dtypes(include=['object']).columns
cat_cols


# In[60]:


# Create dummy variables using the 'get_dummies' command
dummy = pd.get_dummies(df[['Lead Origin', 'Lead Source', 'Specialization',
       'What is your current occupation', 'Search', 'Magazine',
       'Newspaper Article', 'X Education Forums', 'Newspaper',
       'Digital Advertisement', 'Through Recommendations',
       'Receive More Updates About Our Courses',
       'Update me on Supply Chain Content', 'Get updates on DM Content']], drop_first=True) 
                               
# Add the results to the master dataframe
df = pd.concat([df, dummy], axis=1)


# In[61]:


# Drop the variables for which the dummy variables have been created

df = df.drop(['Lead Origin', 'Lead Source','Specialization','Magazine','Newspaper','Digital Advertisement','What is your current occupation'], 1) 


# In[62]:


# Let's take a look at the dataset again

df.head()


# Notice that when you got the value counts of all the columns, there were a few columns in which only one value was majorly present for all the data points. These include Do Not Call, Search, Magazine, Newspaper Article, X Education Forums, Newspaper, Digital Advertisement, Through Recommendations, Receive More Updates About Our Courses, Update me on Supply Chain Content, Get updates on DM Content, I agree to pay the amount through cheque. As we can see all of the values for these variables are No, it's best that we drop these columns as they won't help with our analysis.

# In[63]:


df.drop(['Do Not Call','Search','Newspaper Article','X Education Forums','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content'],axis=1,inplace=True)


# In[64]:


df.head()


# In[65]:


df.shape


# # Test-Train Split

# In[66]:


# Import the required library

from sklearn.model_selection import train_test_split


# In[67]:


# Put all the feature variables in X

# Putting target variable in y
y = df['Converted']

y.head()

X=df.drop('Converted', axis=1)
X.head()


# In[68]:


# Put the target variable in y

y = df['Converted']

y.head()


# In[69]:


# Split the dataset into 70% train and 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# Feature Scaling

# There are a few numeric variables present in the dataset which have different scales, So we can scale these variables.

# In[70]:


# Import MinMax scaler

from sklearn.preprocessing import MinMaxScaler


# In[71]:


# Scale the three numeric features present in the dataset

scaler = MinMaxScaler()

X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])

X_train.head()


# In[72]:


# Checking the Churn Rate
Converted = (sum(df['Converted'])/len(df['Converted'].index))*100
Converted


# The conversion rate is 38.3%

# In[74]:


# Let's see the correlation matrix for total data set
plt.figure(figsize = (20,10))       
sns.heatmap(df.corr(),annot = True, cmap="YlGnBu")
plt.show()


# looking at the above correlaation matrix we can not find any insights

# Model Building

# There are a lot of variables present in the dataset for that we cannot deal with all. So thet the best way is to approach this, is to select a small set of features from this total data of variables using RFE method.

# using X_train and Y_train data

# In[75]:


# Import 'LogisticRegression' and create a LogisticRegression object

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[78]:


# Import RFE and select 15 variables

from sklearn.feature_selection import RFE
logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=15)            
rfe = rfe.fit(X_train,y_train)


# In[79]:


# Let's take a look at which features have been selected by RFE

list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[80]:


# Put all the columns selected by RFE in the variable 'col'

col = X_train.columns[rfe.support_]
col


# In[81]:


X_train.columns[~rfe.support_]


# Assessing the model with statsmodel

# In[82]:


# Import statsmodels

import statsmodels.api as sm


# In[83]:


# Select only the columns selected by RFE

X_train = X_train[col]


# Model 1

# In[84]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# There are quite a few variable which have a p-value greater than 0.05. We will need to take care of them. But first, let's also look at the VIFs.

# In[85]:


# Import 'variance_inflation_factor'

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[86]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# VIFs seem to be in a decent range except for three variables.

# Let's first drop the variable Lead Source_Reference since it has a high p-value as well as a high VIF.

# In[87]:


X_train.drop('Lead Source_Reference', axis = 1, inplace = True)


# # Model 2

# In[88]:


# Refit the model with the new set of features
logm2 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm2.fit().summary()


# In[89]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# The VIFs are now all less than 5. So let's drop the ones with the high p-values beginning with What is your current occupation_Housewife

# In[90]:


# Droping `What is your current occupation_Housewife`.
X_train.drop('What is your current occupation_Housewife', axis = 1, inplace = True)


# # Model 3

# In[91]:


# Refit the model with the new set of features

logm3 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm3.fit().summary()


# The VIFs are now all less than 5. So let's drop the ones with the high p-values beginning with Lead Source_google

# In[92]:


# The column Lead Source_google seems to have high p-value
X_train.drop(['Lead Source_google'],1,inplace=True)


# # Model 4

# In[93]:


# Refit the model with the new set of features

logm4 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm4.fit().summary()


# In[94]:


# The column Lead Source_Olark Chat seems to have a high p-value
X_train.drop(['Specialization_Retail Management'],1,inplace=True)


# # Model 5

# In[95]:


# Refit the model with the new set of features

logm5 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[96]:


# The column What is your current occupation_Working Professional seems to have a high p-value
X_train.drop(['What is your current occupation_Student'],1,inplace=True)


# # Model 6

# In[97]:


# Refit the model with the new set of features

logm6 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm6.fit().summary()


# In[98]:


# The column What is your current occupation_Other seems to have a high p-value
X_train.drop(['Newspaper_Yes'],1,inplace=True)


# # Model 7

# In[99]:


# Refit the model with the new set of features

logm7 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
res=logm7.fit()
res.summary()


# All the p-values are now in the appropriate range(0.05). Let's also check the VIFs again in case we had missed something.
# 
# 

# In[100]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Both the p-values and VIFs seem to be decent enough for all the variables. So let's make predictions using this final set of features.

# In[101]:


# Predicting the train variables
X_train_sm = sm.add_constant(X_train)
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[102]:


# Reshaping it into an array

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:9]


# Creating a dataframe with the actual churn flag and the predicted probabilities

# In[103]:


# Create a new dataframe containing the actual churn flag and the probabilities predicted by the model

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# Creating new column 'Predicted' with 1 if Paid_Prob > 0.5 else 0

# In[104]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# Till now that we have the probabilities and we have also made some conversion predictions using them, by using all these we can eveluate the model

# # Model Evaluation

# In[105]:


# Import metrics from sklearn for evaluation

from sklearn import metrics


# In[106]:


# Create confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# Accuracy

# In[107]:


# Let's check the overall accuracy

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[108]:


# Let's evaluate the other metrics as well

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# sensitivity

# In[109]:


# Calculate the sensitivity

TP/(TP+FN)


# specificity

# In[110]:


# Calculate the specificity

TN/(TN+FP)


# # Finding the Optimal Cutoff

# Now 0.5 was just arbitrary to loosely check the model performace. But in order to get good results, you need to optimise the threshold. So first let's plot an ROC curve to see what AUC we get.

# In[111]:


# ROC function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[112]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[113]:


import matplotlib.pyplot as plt


# In[114]:


# Call the ROC function

draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# The area under the curve of the ROC is 0.84 which is quite good. So we seem to have a good model. Let's also check the sensitivity and specificity tradeoff to find the optimal cutoff point.

# In[116]:


# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[117]:


# Let's create a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[118]:


# Let's plot it as well

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# As you can see that around 0.3, you get the optimal values of the three metrics. So let's choose 0.3 as our cutoff now.

# In[119]:


#### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.
y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# Accuracy

# In[120]:


# Let's check the accuracy now

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[121]:


# Let's create the confusion matrix once again

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[122]:


# Let's evaluate the other metrics as well

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# Sensitivity

# In[123]:


# Calculate Sensitivity

TP/(TP+FN)


# Specificity

# In[124]:


# Calculate Specificity

TN/(TN+FP)


# The cutoff point 0.3 seems to be good.

# # Making Predictions

# Now make predicitons on the train set.

# In[125]:


# Scale the test set as well using just 'transform'

X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[126]:


# Select the columns in X_train for X_test as well

X_test = X_test[col]
X_test.head()


# In[127]:


# Add a constant to X_test

X_test_sm = sm.add_constant(X_test[col])


# In[128]:


# Check X_test_sm

X_test_sm


# In[129]:


# Make predictions on the test set and store it in the variable 'y_test_pred'
X_test =X_test[X_train.columns]
y_test_pred = res.predict(sm.add_constant(X_test))


# In[130]:


y_test_pred[:10]


# In[131]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)


# In[132]:


# Let's see the head

y_pred_1.head()


# In[133]:


# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)


# In[134]:


# Remove index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[135]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[136]:


# Check 'y_pred_final'

y_pred_final.head()


# In[137]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})


# In[138]:


# Let's see the head of y_pred_final

y_pred_final.head()


# In[139]:


# Make predictions on the test set using 0.3 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.3 else 0)


# In[140]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[141]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# sensitivity

# In[142]:


# Calculate sensitivity
TP / float(TP+FN)


# Specificity

# In[144]:


# Calculate specificity
TN / float(TN+FP)


# # Precision and Recall View

# Build the training model using the precision and recall view

# In[145]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# Precision

# TP / TP + FP

# In[146]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# Recall

# TP / TP + FN

# In[147]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# Precision and recall tradeoff

# In[148]:


from sklearn.metrics import precision_recall_curve


# In[149]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[150]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[151]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# Threshold point is 0.3

# In[152]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# Accuracy

# In[153]:


# Let's check the accuracy now

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[154]:


# Let's create the confusion matrix once again

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[155]:


# Let's evaluate the other metrics as well

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# Precision

# In[156]:


# Calculate Precision

TP/(TP+FP)


# Recall

# In[157]:


# Calculate Recall

TP/(TP+FN)


# This cutoff point is very good

# In[158]:


X_test.shape


# In[159]:


X_train.shape


# In[160]:


X_train.columns


# In[161]:


X_test.columns


# In[162]:


X_test =X_test[X_train.columns]


# Now make predicitons on the test set.

# In[163]:


# Make predictions on the test set and store it in the variable 'y_test_pred'
y_test_pred = res.predict(sm.add_constant(X_test))
y_test_pred[:10]


# In[164]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1.head()


# In[165]:


# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
y_test_df.head()


# In[166]:


# Remove index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[167]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[168]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[169]:


# Make predictions on the test set using 0.3 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.3 else 0)
y_pred_final.head()


# Accuracy

# In[170]:


# Let's check the overall accuracy

metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[171]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[172]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# Precision

# In[173]:


# Calculate Precision

TP/(TP+FP)


# Recall

# In[174]:


# Calculate Recall

TP/(TP+FN)


# Final Observation:

# Let us compare the values obtained for Train & Test:

# Train Data:
# 
# Accuracy : 78%
# Sensitivity :76%
# Specificity : 80%
# Precision :72%
# Recall :76%

# Test Data:
# 
# Accuracy : 80%
# Sensitivity : 70%
# Specificity : 86%
# Precision : 74%
# Recall : 69%

# With the current cut off as 0.3 we have Precision around 73% and Recall around 72%

# The Model seems to predict the Conversion Rate very well and we should be able to give the confidence in making good calls based on this model

# In[ ]:




