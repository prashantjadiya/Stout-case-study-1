#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd


# In[10]:


df=pd.read_csv("loans_full_schema.csv")


# In[11]:


#What is the common purpose of all of the loan takers?

import numpy as np
import matplotlib.pyplot as plt
x_bar=df['loan_purpose']


# In[12]:


df['loan_purpose'].value_counts().plot(kind='barh')


# In[ ]:





# In[13]:


rslt_df = df['account_never_delinq_percent'][df['homeownership']=='OWN'] 
rslt_df.describe()


# In[16]:


#how the delinqu percentage is related with homeownership

# rslt_df=pd.cut(rslt_df,bins=[40,60,80,95,100],labels=['60%','40%','20%','5%'])
#delinque rate for homeowners who have their OWN house

rslt_df.value_counts().plot(kind='bar',title="Delinque rate for homeowners who have their OWN house",xlabel="Delinque rate",ylabel='homewowners count')


# In[ ]:





# In[ ]:





# In[24]:


#which state has more public record bankrupts

x=df[['state','public_record_bankrupt']]
x.groupby(['state']).count().sort_values(by=['public_record_bankrupt']).plot(kind="bar",title="Bankrupts in public record",ylabel="number of bankrupts",xlabel="state",figsize=(10,10))


# In[ ]:


#highest bankrupts were in California


# In[ ]:





# In[ ]:


#how many have paid interest fees if they are late in Loan payment


# In[25]:


df['loan_status'].value_counts()


# In[26]:


(df['paid_late_fees']!=0.00).sum()


# In[29]:


import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Paid','Not paid'
sizes = [52,54]
explode = (0.1,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("How many have paid interest fees if they are late in Loan payment")
plt.show()


# In[30]:


tcl=df['total_credit_limit']/1000

import seaborn as sns
sns.boxplot(tcl)


# In[31]:


ain=df['annual_income']/1000

sns.boxplot(ain)


# In[32]:


ratio=tcl/ain
ratio


# In[33]:


ratio.value_counts()


# In[34]:


type(ratio)


# In[35]:


rslt_df2=pd.cut(ratio,bins=[0,1,2,3,4,5,6,7,8],labels=['1x','2x','3x','4x','5x','6x','7x','8x'])


# In[36]:


rslt_df2.value_counts().plot(kind="bar",title="Total credit limit to annual income ratio",xlabel="Credit limit (Multiplier of Annual income)",ylabel="Number of people",figsize=(8,8))


# In[ ]:





# In[ ]:





# In[ ]:




