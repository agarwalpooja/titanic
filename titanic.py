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
df = td_not_survived.groupby('Pclass').size()
plt.figure();df.plot(kind='bar').set_title('Not Survived passengers by Pclass')