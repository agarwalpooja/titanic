import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading Training and Testing Data =====>")
training_data = pd.read_csv(r'C:\Users\ASUS\Documents\jbg_ml\titanic\train.csv')
testing_data = pd.read_csv(r'C:\Users\ASUS\Documents\jbg_ml\titanic\test.csv')
print("<===== Training and Testing Data Loading finished")

training_data.describe()
testing_data.describe()
training_data.head()
testing_data.head()
training_data.sample(10)
training_data.dtypes


td_not_survived=training_data.loc[(training_data['Survived']==0)]
td_survived=training_data.loc[(training_data['Survived']==1)]
td_not_survived.head(5)
td_survived.sample(10)
#plot
f,ax = plt.subplots(3,4,figsize=(20,16))
sns.countplot('Pclass',data=training_data,ax=ax[0,0])
sns.countplot('Sex',data=training_data,ax=ax[0,1])
sns.boxplot(x='Pclass',y='Age',data=training_data,ax=ax[0,2])
sns.countplot('SibSp',hue='Survived',data=training_data,ax=ax[0,3],palette='husl')
sns.distplot(training_data['Fare'].dropna(),ax=ax[2,0],kde=False,color='b')
sns.countplot('Embarked',data=training_data,ax=ax[2,2])

sns.countplot('Pclass',hue='Survived',data=training_data,ax=ax[1,0],palette='husl')
sns.countplot('Sex',hue='Survived',data=training_data,ax=ax[1,1],palette='husl')
sns.distplot(training_data[training_data['Survived']==0]['Age'].dropna(),ax=ax[1,2],kde=False,color='r',bins=5)
sns.distplot(training_data[training_data['Survived']==1]['Age'].dropna(),ax=ax[1,2],kde=False,color='g',bins=5)
sns.countplot('Parch',hue='Survived',data=training_data,ax=ax[1,3],palette='husl')
sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=training_data,palette='husl',ax=ax[2,1])
sns.countplot('Embarked',hue='Survived',data=training_data,ax=ax[2,3],palette='husl')

ax[0,0].set_title('Total Passengers by Class')
ax[0,1].set_title('Total Passengers by Gender')
ax[0,2].set_title('Age Box Plot By Class')
ax[0,3].set_title('Survival Rate by SibSp')
ax[1,0].set_title('Survival Rate by Class')
ax[1,1].set_title('Survival Rate by Gender')
ax[1,2].set_title('Survival Rate by Age')
ax[1,3].set_title('Survival Rate by Parch')
ax[2,0].set_title('Fare Distribution')
ax[2,1].set_title('Survival Rate by Fare and Pclass')
ax[2,2].set_title('Total Passengers by Embarked')
ax[2,3].set_title('Survival Rate by Embarked')
#group by gender 
df = training_data.groupby(['Sex','Survived']).size() # output pandas.core.series.Series
type(df) # pandas.core.series.Series
#df=df.unstack()
df.head()

plt.figure();df.plot(kind='bar').set_title('Gender histogram training data')

df=df.unstack()
plt.figure();df.plot(kind='bar').set_title('Gender histogram training data')
#survived n sex
df = td_survived.groupby('Sex').size()
#df=df.unstack()
df.head()
plt.figure();df.plot(kind='bar').set_title('Survived passengers by gender')
#noy survived vs sex
df = td_not_survived.groupby('Sex').size()
plt.figure();df.plot(kind='bar').set_title(' Not Survived passengers by gender')

#2.3.3 Plotting histogram of survived by Pclass
df = td_survived.groupby('Pclass').size()
plt.figure();df.plot(kind='bar').set_title('Survived passengers by Pclass')
df = td_not_survived.groupby('Pclass').size()
plt.figure();df.plot(kind='bar').set_title('Not Survived passengers by Pclass')

#2.3.4 Plotting histogram of survived by Age
plt.figure()
td_survived.Age.hist()
plt.figure()
plt.suptitle("Passengers Age distribution",x=0.5, y=1.05, ha='center', fontsize='xx-large')
pl1 = training_data.Age.hist()
pl1.set_xlabel("Age")
pl1.set_ylabel("Count")

df_children = training_data.loc[(training_data['Age']>=0) & (training_data['Age']<=15)]
df_y_adults = training_data.loc[(training_data['Age'] >15)].loc[(training_data['Age']<=30 )]
df_adults = training_data.loc[(training_data['Age'] >30)].loc[(training_data['Age']<=60 )]
df_old = training_data.loc[(training_data['Age'] >60)]


plt.figure()
df1 = df_children.groupby('Survived').size() # with .size() we generate a pandas pandas.core.series.Series Series type variable
plt.subplot(2,2,1)
df1.plot(kind='bar').set_title('Children') 
df2 = df_y_adults.groupby('Survived').size() # with .size() we generate a pandas  pandas.core.series.Series Series type variable
plt.subplot(2,2,2)
df2.plot(kind='bar').set_title('young Adults')
df3 = df_adults.groupby('Survived').size() # with .size() we generate a pandas pandas.core.series.Series Series type variable
plt.subplot(2,2,3)
df3.plot(kind='bar').set_title('Adults')
df4 = df_old.groupby('Survived').size() # with .size() we generate a pandas pandas.core.series.Series Series type variable
plt.subplot(2,2,4)
df4.plot(kind='bar').set_title('old')

f,ax = plt.subplots(2,2,figsize=(10,10))
sns.countplot('Survived',data=df_children,ax=ax[0,0])
sns.countplot('Survived',data=df_y_adults,ax=ax[0,1])
sns.countplot('Survived',data=df_adults,ax=ax[1,0])
sns.countplot('Survived',data=df_old,ax=ax[1,1])

ax[0,0].set_title('Survival Rate by children')
ax[0,1].set_title('Survival Rate by young adults')
ax[1,0].set_title('Survival Rate by adults')
ax[1,1].set_title('Survival Rate by old')

#analysing data
df_full = pd.concat([training_data,testing_data], sort=True) # axis : {0/’index’, 1/’columns’}, default 0 The axis to concatenate along (by index)
num_all = len(df_full.index)
''' number of records of training data'''
num_train = len(training_data.index)
''' number of records of testing data'''
num_test = len(testing_data.index)
d = {'full' : num_all, 'train' : num_train, 'test' : num_test}
number_records = pd.Series(d)

number_records.head()

df_sum_null = df_full.isnull().sum().sort_values(ascending=False) # output pandas.core.series.Series
#df=df_sum_null.unstack() ==> does not work
plt.figure();df_sum_null.plot(kind='barh') # showing a horizontal bar plot 