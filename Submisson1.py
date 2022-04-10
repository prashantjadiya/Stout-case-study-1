#!/usr/bin/env python
# coding: utf-8
Case Study #1
Below is a data set that represents thousands of loans made through the Lending Club platform, which is a platform that allows individuals to lend to other individuals.
We would like you to perform the following using the language of your choice:

•	Describe the dataset and any issues with it.
•	Generate a minimum of 5 unique visualizations using the data and write a brief description of your observations. Additionally, all attempts should be made to make the visualizations visually appealing
•	Create a feature set and create a model which predicts interest_rate using at least 2 algorithms. Describe any data cleansing that must be performed and analysis when examining the data.
•	Visualize the test results and propose enhancements to the model, what would you do if you had more time. Also describe assumptions you made and your approach.

Dataset
https://www.openintro.org/data/index.php?data=loans_full_schema 

Output
An HTML website hosting all visualizations and documenting all visualizations and descriptions. All code hosted on GitHub for viewing. Please provide URL’s to both the output and the GitHub repo.

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder 


# In[2]:


df=pd.read_csv("loans_full_schema.csv")
df


# In[3]:


df.describe()


# In[4]:


df.isna().sum()


# In[5]:


df.dtypes

Will go through each column to clean the data
# In[6]:


x=df['emp_title'].unique()
len(x)


# In[7]:


#not useful column so will drop it


# In[8]:


df=df.drop(columns=['emp_title'])


# In[9]:


#second
df['emp_length'].isna().sum()


# In[10]:


df['emp_length'].unique()


# In[11]:


df['emp_length'].interpolate(inplace=True)
#simple imputer as linear


# In[12]:


df['emp_length'].unique()


# In[ ]:





# In[13]:


len(df['state'].unique()) #states are correct as per the record


# In[14]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
df['state'] = le.fit_transform(df['state'])


# In[ ]:





# In[15]:


#homeownership
le = LabelEncoder() 
df['homeownership'] = le.fit_transform(df['homeownership'])


# In[ ]:





# In[16]:


#annual_income
df['annual_income']=df['annual_income'].astype(int)


# In[ ]:





# In[17]:


df['verified_income'].unique()


# In[18]:


#verified_income
le = LabelEncoder() 
df['verified_income'] = le.fit_transform(df['verified_income'])


# In[ ]:





# In[19]:


#debt_to_income
df.debt_to_income.interpolate(inplace=True)


# In[20]:


df.debt_to_income.isna().sum()


# In[ ]:





# In[21]:


df.annual_income_joint.interpolate(inplace=True)


# In[22]:


df.annual_income_joint.isna().sum()


# In[23]:


df['annual_income_joint'].fillna((df['annual_income_joint'].mean()), inplace=True)
#peli char values ne mean thi krvi pdi impute


# In[ ]:





# In[ ]:





# In[24]:


df.verification_income_joint.isna().sum()


# In[25]:


df.verification_income_joint.interpolate(inplace=True,method="ffill")


# In[26]:


df.verification_income_joint.isna().sum()


# In[27]:


df.verification_income_joint.unique()


# In[28]:


df.verification_income_joint = df.verification_income_joint.fillna(df.verification_income_joint.mode().iloc[0])


# In[29]:


df.verification_income_joint.isna().sum()


# In[30]:


le = LabelEncoder() 
df['verification_income_joint'] = le.fit_transform(df['verification_income_joint'])


# In[ ]:





# In[31]:


df.debt_to_income_joint.interpolate(inplace=True)


# In[32]:


df.debt_to_income_joint.isna().sum()


# In[33]:


df['debt_to_income_joint'].fillna((df['debt_to_income_joint'].mean()), inplace=True)
#peli char values ne mean thi krvi pdi impute


# In[34]:


df.debt_to_income_joint.isna().sum()


# In[ ]:





# In[35]:


df.months_since_last_delinq.interpolate(inplace=True)


# In[36]:


df.months_since_last_delinq.isna().sum()


# In[37]:


df['months_since_last_delinq']=df['months_since_last_delinq'].astype(int)


# In[38]:


df['months_since_last_delinq'].dtype


# In[ ]:





# In[39]:


df.inquiries_last_12m.unique()


# In[40]:


df.total_credit_lines.unique()


# In[41]:


df.num_collections_last_12m.unique()


# In[42]:


df.num_historical_failed_to_pay.unique()


# In[ ]:





# In[ ]:





# In[43]:


df.months_since_90d_late.isna().sum()


# In[44]:


df.months_since_90d_late.interpolate(inplace=True)


# In[45]:


df.months_since_90d_late.isna().sum()


# In[ ]:





# In[46]:


df.months_since_last_credit_inquiry.isna().sum()


# In[47]:


df.months_since_last_credit_inquiry.interpolate(inplace=True)


# In[48]:


df['months_since_last_credit_inquiry']=df['months_since_last_credit_inquiry'].astype(int)


# In[ ]:





# In[49]:


df.num_accounts_120d_past_due.isna().sum()


# In[50]:


df.num_accounts_120d_past_due.interpolate(inplace=True)


# In[51]:


df['num_accounts_120d_past_due']=df['num_accounts_120d_past_due'].astype(int)


# In[ ]:





# In[ ]:





# In[52]:


df['loan_purpose'].unique()


# In[53]:


df['application_type'].unique()


# In[54]:


le = LabelEncoder() 
df['application_type'] = le.fit_transform(df['application_type'])


# In[55]:


df['application_type']=df['application_type'].astype(int)


# In[ ]:





# In[56]:


df['grade'].unique()


# In[57]:


le = LabelEncoder() 
df['grade'] = le.fit_transform(df['grade'])
df['grade']=df['grade'].astype(int)


# In[ ]:





# In[58]:


df['sub_grade'].unique()


# In[59]:


le = LabelEncoder() 
df['sub_grade'] = le.fit_transform(df['sub_grade'])
df['sub_grade']=df['sub_grade'].astype(int)


# In[ ]:





# In[60]:


df['issue_month'].unique()


# In[61]:


le = LabelEncoder() 
df['issue_month'] = le.fit_transform(df['issue_month'])
df['issue_month']=df['issue_month'].astype(int)


# In[ ]:





# In[62]:



le = LabelEncoder() 
df['loan_purpose'] = le.fit_transform(df['loan_purpose'])
df['loan_purpose']=df['loan_purpose'].astype(int)


# In[ ]:





# In[63]:


df['loan_status'].unique()


# In[64]:


le = LabelEncoder() 
df['loan_status'] = le.fit_transform(df['loan_status'])
df['loan_status']=df['loan_status'].astype(int)


# In[ ]:





# In[65]:


df['initial_listing_status'].unique()


# In[66]:


le = LabelEncoder() 
df['initial_listing_status'] = le.fit_transform(df['initial_listing_status'])
df['initial_listing_status']=df['initial_listing_status'].astype(int)


# In[ ]:





# In[67]:


df['disbursement_method'].unique()


# In[68]:


le = LabelEncoder() 
df['disbursement_method'] = le.fit_transform(df['disbursement_method'])
df['disbursement_method']=df['disbursement_method'].astype(int)


# In[ ]:





# In[ ]:





# In[69]:


df.isna().sum()


# In[70]:


df.dtypes


# In[ ]:





# In[71]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[72]:


c = df.corr().abs()
s = c.unstack()
so = s.sort_values(kind="quicksort")
so


# In[73]:


x=df.corr().unstack()
x


# In[ ]:





# In[ ]:





# In[74]:


X=df.loc[:, df.columns != 'interest_rate']
y=df['interest_rate']

from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
linear_regression= LinearRegression()
linear_regression.fit(X_train,y_train)
print('Accuracy on test set: {:.2f}'.format(linear_regression.score(X_test, y_test)))
from sklearn.metrics import mean_squared_error
y_pred=linear_regression.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("LR MSE ",mse)

from sklearn.tree import DecisionTreeRegressor
dtr= DecisionTreeRegressor()
dtr.fit(X_train,y_train)
y_pred=dtr.predict(X_test)
print('Accuracy on test set: {:.2f}'.format(dtr.score(X_test, y_test)))
mse = mean_squared_error(y_test, y_pred)
print("DTR MSE ",mse)

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=20,max_depth=10,criterion='mse',)
forest.fit(X_train,y_train)
y_pred=forest.predict(X_test)
print('Accuracy on test set: {:.2f}'.format(forest.score(X_test, y_test)))
mse = mean_squared_error(y_test, y_pred)
print("RF MSE ",mse)

from sklearn.svm import SVR
svr_rbf=SVR(C=1.0, epsilon=0.01, kernel='rbf')
svr_rbf.fit(X_train, y_train)
y_pred = svr_rbf.predict(X_test)
print('Accuracy on test set: {:.2f}'.format(svr_rbf.score(X_test, y_test)))
mse = mean_squared_error(y_test, y_pred)
print("SVR MSE ",mse)

from sklearn.linear_model import RANSACRegressor
reg = RANSACRegressor(max_trials=100,residual_threshold=4,min_samples=16,random_state=0).fit(X_train, y_train)
y_pred=reg.predict(X_test)
print('Accuracy on test set: {:.2f}'.format(reg.score(X_test, y_test)))
mse = mean_squared_error(y_test, y_pred)
print("RANSAC MSE ",mse)



# In[75]:


features=x['interest_rate'].sort_values()
features


# In[76]:


X=df[['total_debit_limit','disbursement_method','num_mort_accounts','total_credit_limit','account_never_delinq_percent','initial_listing_status','annual_income','months_since_last_credit_inquiry','num_total_cc_accounts','total_credit_lines','earliest_credit_line','loan_status','accounts_opened_24m','inquiries_last_12m','debt_to_income','verified_income','term','paid_interest']]


# In[77]:


y=df['interest_rate']


# In[78]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

linear_regression= LinearRegression()
linear_regression.fit(X_train,y_train)
print('Accuracy on test set: {:.2f}'.format(linear_regression.score(X_test, y_test)))
y_pred=linear_regression.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("LR MSE ",mse)

dtr= DecisionTreeRegressor()
dtr.fit(X_train,y_train)
y_pred=dtr.predict(X_test)
print('Accuracy on test set: {:.2f}'.format(dtr.score(X_test, y_test)))
mse = mean_squared_error(y_test, y_pred)
print("DTR MSE ",mse)

forest=RandomForestRegressor(n_estimators=20,max_depth=10,criterion='mse',)
forest.fit(X_train,y_train)
y_pred=forest.predict(X_test)
print('Accuracy on test set: {:.2f}'.format(forest.score(X_test, y_test)))
mse = mean_squared_error(y_test, y_pred)
print("RF MSE ",mse)

svr_rbf=SVR(C=1.0, epsilon=0.01, kernel='rbf')
svr_rbf.fit(X_train, y_train)
y_pred = svr_rbf.predict(X_test)
print('Accuracy on test set: {:.2f}'.format(svr_rbf.score(X_test, y_test)))
mse = mean_squared_error(y_test, y_pred)
print("SVR MSE ",mse)

reg = RANSACRegressor(max_trials=100,residual_threshold=4,min_samples=16,random_state=0).fit(X_train, y_train)
y_pred=reg.predict(X_test)
print('Accuracy on test set: {:.2f}'.format(reg.score(X_test, y_test)))
mse = mean_squared_error(y_test, y_pred)
print("RANSAC MSE ",mse)


# In[ ]:




•	Describe the dataset and any issues with it. Merged with Data Cleansing part


Dataset is of the loans made through the Lending Club Platfrom as described in the case study.
Dataset has many issues as in terms of Null vales, outliers etc.

Null values are handled by interpolate and SimpleImputer method which goes for emp_length, debt_to_income, annual_income_joint, verification_income_joint, debt_to_income_joint,  months_since_last_delinq,  months_since_90d_late,
months_since_last_credit_inquiry,  num_accounts_120d_past_due.

First of all, I checked the null values in each column, then validated the datatype and the values. For example, I have checked whether emp_length feature contains any non-numerical value or not.

Encoded the categorical features to numerical numbers. ex: Male, female to 0,1

Special cases for some features like annual_income_joint, I have to fill null values with interpolate method, but the first four null values couldn't be filled with it, so I have to fill them with mean of all values. The same goes for debt_to_income feature.

Similarly, I had to fill first four values of verification_income_joint with most frequent value in the column.

Then converted all features' datatypes to correct ones like if there are not any float values in the column, then it should be transformed to int datatype.

•	Generate a minimum of 5 unique visualizations using the data and write a brief description of your observations. Additionally, all attempts should be made to make the visualizations visually appealing

is in another notebook
# In[ ]:





# In[79]:


#shap values

import shap
explainer = shap.Explainer(forest)
shap_values = explainer(X_test)


# In[ ]:





# In[80]:


shap.plots.heatmap(shap_values[1:100])


# In[81]:



shap.plots.bar(shap_values[0])


# In[82]:


shap_values = explainer.shap_values(X_test)
shap.initjs()
def p(j):
    return(shap.force_plot(explainer.expected_value, shap_values[j,:], X_test.iloc[j,:]))


# In[83]:


p(10)


# In[84]:


expected_value = explainer.expected_value
print("The expected value is ", expected_value)
shap_values = explainer.shap_values(X_test)[0]
shap.decision_plot(expected_value, shap_values, X_test)


# In[ ]:




