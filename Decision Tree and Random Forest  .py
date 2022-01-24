#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[9]:


#**Import Data set** 


# In[10]:


df=pd.read_csv("Titanic_passengers.csv")


# In[11]:


df.head()


# In[12]:


df.shape


# In[14]:


#**Clean the data
df.info()


# In[15]:


df.isnull().sum()


# In[16]:


#**Replacing Missing Numerical Values with the appropriate values**
#Replacing Missing Numerical Values of the "Age" Feature


# In[17]:


df["Age"].fillna(df["Age"].mean(), inplace= True)


# In[18]:


#**Replacing Missing Categorical Values with the appropriate values**
#Number of Elements per Category of the "Cabin","Fare" Feature


# In[19]:


df["Cabin"].fillna(df["Cabin"].mode()[0],inplace=True)


# In[24]:


print(df["Cabin"].value_counts())


# In[22]:


df["Fare"].fillna(df["Fare"].mode()[0],inplace=True)


# In[25]:


print(df["Fare"].value_counts())


# In[26]:


print(df["Embarked"].value_counts())


# In[23]:


df.isnull().sum()


# # Visualisation Phase

# # Distribution of the most important features
# #**"Age" Feature Distribution**

# In[31]:


import matplotlib.pyplot as plt 
plt.title ("Histogram of Different Ages")
plt.xlabel("Age")
df["Age"].plot.hist()


# **"Pclass"Feature Distribution**

# In[36]:


import seaborn as sns 


# In[37]:


sns.countplot(x= "Pclass", data= df)
plt.xticks(rotation=-45)


# Majority of passengers are in 3rd class

# **"SEX" Distribution**

# In[38]:


sns.countplot(x= "Sex", data= df)
plt.xticks(rotation=-45)


# **"Embarked" Feature Distribution**

# In[39]:


sns.countplot(x= "Embarked",data= df)
plt.xticks(rotation=-45)


#  Majority of passengers embarked at S (Southampton).

# # Correlation between "Sex" and "Age"

# In[40]:


g=sns.FacetGrid(df, row= "Survived", col="Sex")
g.map(plt.hist, "Age", bins=20)


# # Correlation between "Pclass" and "Embarked"

# In[41]:


grid=sns.FacetGrid(df, row= "Survived", col="Pclass")
grid.map(plt.hist, "Embarked", bins=20)


# # Correlation Heatmap

# In[42]:


def plot_correlation_map( df ):

    corr = df.corr()

    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }
        
    )

plot_correlation_map(df)


# **To start with, the number of male passengers we extracted from the "Sex" feature distribution is much higher than the number of female passengers. From the correlation between "Age" and "Sex" features above, we can deduce that females are more likely to survive than males. Also, according to the "Age" feature distribution, most passengers are in 15-35 years range. However, a large number of passengers that are between 15 and 25 years old did not survive the incident. We then visualized the correlation between "Pclass" and "Embarked" features which supplied us with the information that the 3rd class passengers are more likely to be dead than other classes and 1st class passengers are more likely to survive than other classes. In addition, passengers who embarked at C had a higher survival rate than people who embarked at S or Q. Last but not least, the function we defined was used to build a correlation heatmap which is, in fact, a graphical representation of correlation matrix representing correlation between all the different features of our dataset. The value of correlation can take any values between -1 and 1. Values closer to zero means there is no linear relationship between the two variables. The closer to 1 the correlation is, the more positively correlated the variables are. The closer to -1 the correlation is, the more negatively correlated the variables are. For example, the correlation between SibSp and Parch variables is equal to 0.41. They are somehow considered to have a moderate positive relationship which means that parents and siblings like to travel together. Also, the correlation between Pclass and Fare variables is equal to -0.55. They have a strong negative correlation which means that higher economic classes ("Pclasses") pay more for their trip ("Fare").**

# # Correlation grouby between "Pclass" and "Survived"

# In[43]:


cleanup= {"Survived": {"No":0,"Yes":1}}
df.replace(cleanup, inplace=True)
df[["Pclass","Survived"]].groupby(["Pclass"], as_index=True).mean()


# In[44]:


df


# # Dropping useless columns 

# In[90]:


new_df= df.copy()
new_df= new_df.drop(["PassengerId", "Name", "Ticket", "Cabin",], axis=1)
new_df.head()


# # Create a new Feature "FamilySize" from "SibSp" and "Parch" Features

# In[91]:


new_df["FamilySize"]= new_df["SibSp"]+new_df["Parch"]
new_df= new_df.drop("SibSp", axis= 1)
new_df= new_df.drop("Parch", axis= 1)
new_df.head()


# # Create a new column "Title"

# In[92]:


new_df['Title'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
new_df.head()


# # Number of Elements per Category of the "Title" Feature

# In[93]:


print(new_df["Title"].value_counts())


# **Correlation between "Title" and "Age"**

# In[94]:


new_df[["Age", "Title"]].groupby(["Title"], as_index=True).mean()


# **Correlation between "Title" and "Fare"**

# In[95]:


new_df[["Fare", "Title"]].groupby(["Title"], as_index=True).mean()


# **Correlation between "Title" and "Pclass"**

# In[96]:


new_df[["Pclass", "Title"]].groupby(["Title"], as_index=True).mean()


# **Correlation between "Title" and "Sex"**

# In[97]:


g=sns.FacetGrid(new_data, col="Sex")
g.map(plt.hist, "Title", bins=20)


# # Creating a Dictionary

# In[98]:



Title_Dictionary={"Capt": "Officer",
                  
                  "Col": "Officer", 
                  
                  "Major": "Officer",
                  
                  "Dr": "Officer", 
                  
                  "Rev": "Officer",
                  
                  "Jonkheer": "Royalty",
                  
                  "Don": "Royalty", 
                  
                  "Sir": "Royalty",
                  
                  "Lady": "Royalty",
                  
                  "the Countess": "Royalty",
                  
                  "Dona": "Royalty", 
                  
                  "Mme": "Miss",
                  
                  "Mlle": "Miss",
                  
                  "Miss": "Miss",
                  
                  "Ms": "Mrs", 
                  
                  "Mr": "Mrs",
                  
                  "Mrs": "Mrs", 
                  
                  "Master": "Master" 
                 }


# **Add the Dictonary as a column to the DataFrame**

# In[99]:


new_df['Title'] = new_data['Title'].map(Title_Dictionary)
new_df.head()


# **New Number of Elements per Category of the "Title" Feature**

# # Visualize the same correlations
# **Between "Title" and "Age"**

# In[100]:


new_df[["Age", "Title"]].groupby(["Title"], as_index=True).mean()


# **Between "Title" and "Pclass"**

# In[101]:


new_df[["Pclass", "Title"]].groupby(["Title"], as_index=True).mean()


# **Between "Title" and "Sex"**

# In[102]:


g=sns.FacetGrid(new_df, col="Sex")
g.map(plt.hist, "Title", bins=20)


# In[103]:


new_df.head()


# # Correlation between "Survived" and "FamilySize"

# In[104]:


new_df[["Survived", "FamilySize"]].groupby(["FamilySize"], as_index=True).mean()


# In[105]:


g=sns.FacetGrid(new_data, col="Survived")
g.map(plt.hist, "FamilySize", bins=20)


# **"FamilySize" feature is useful as it shows us that survival rates are better when passengers are accompanied rather than alone**

# # Features Transformation

# In[106]:


from sklearn.preprocessing import LabelEncoder


# In[107]:


encoder=LabelEncoder()
new_df["Title"]=encoder.fit_transform(new_df["Title"])
new_df["Embarked"]=encoder.fit_transform(new_df["Embarked"])
new_df["Sex"]=encoder.fit_transform(new_df["Sex"])
new_df


# # Decision Tree

# **Features Extraction**

# In[108]:


X = new_df[["Pclass", "Age", "Sex", "FamilySize", "Fare", "Title"]]
y = new_df["Survived"]


# **Split the dataset into train and test sets**

# In[109]:


from sklearn.model_selection import train_test_split


# In[110]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# **Applying Algorithm**

# In[111]:


from sklearn import tree   
from sklearn.tree import DecisionTreeClassifier
dt = tree.DecisionTreeClassifier()  


# **Fitting training data**

# In[112]:


model= dt.fit(X_train, y_train)


# **Testing model's performance**

# In[113]:


y_pred= model.predict(X_test)


# **Performance of the Decision Tree**

# In[115]:


from sklearn.metrics import accuracy_score


# In[116]:


print("score:{}".format(accuracy_score(y_test, y_pred)))


# # Decision Tree Visualization

# In[122]:


#importing relevant libraries
from sklearn.model_selection import train_test_split
from sklearn import tree   
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


import graphviz
dot_data= tree.export_graphviz(model , out_file=None)
graph= graphviz.Source(dot_data)
graph.render("new_df")
graph


# *The first line of each node (except those of the final row) shows the splitting condition in the form "feature <= value". Next, we find the Gini Impurity of the node, which quantifies the purity of the node/leaf. A Gini score greater than zero implies that samples contained within that node belong to different classes. "Samples" is simply the number of observations contained in the node. Lastly, "Value" shows the class distribution of the samples ([count non_survived, count survived]).*
# 
# *The split that costs least is chosen since we always want to maximize our accuracy. In our example, the best split is the sex of the passenger. The tree first splits by sex, and then by class, since it has learned during the training phase that these are the two most important features for determining survival. Interestingly, after splitting by class, the main deciding factor determining the survival of women is the ticket fare that they paid, while the deciding factor for men is their age (with children being much more likely to survive).*

# # Changing Decision Tree Parameters

# In[126]:


dtree= tree.DecisionTreeClassifier(criterion= "gini", splitter= "random",  max_leaf_nodes= 10, min_samples_leaf= 5, max_depth=10)


# In[127]:


model1= dtree.fit(X_train, y_train)


# In[128]:


y_pred= model1.predict(X_test)


# In[129]:


print("score:{}".format(accuracy_score(y_test, y_pred)))


# After changing some of the decision tree parameters , our overall accuracy was still 1%. 

# In[ ]:


dot_data= tree.export_graphviz(model1 , out_file=None)
graph= graphviz.Source(dot_data)
graph.render("new_df")
graph


# # Random Forest
# **Creating a Random Forest**

# In[131]:


from sklearn.ensemble import RandomForestClassifier


# In[132]:


clf= RandomForestClassifier()


# # Training our Model

# In[133]:


clf.fit(X_train, y_train) 


# # Testing our Model

# In[134]:


y_pred=clf.predict(X_test)


# # Testing our Model's accuracy

# In[135]:


from sklearn import metrics 


# In[136]:


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[137]:


clf1= RandomForestClassifier(n_estimators=50)


# In[138]:


clf1.fit(X_train, y_train) 


# In[139]:


y_pred=clf1.predict(X_test)


# In[140]:


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# By changing the number estimators, the result shows that our random forest classifier's overall accuracy remained the same 

# In[ ]:




