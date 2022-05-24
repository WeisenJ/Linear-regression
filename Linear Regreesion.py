#!/usr/bin/env python
# coding: utf-8

# In[24]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


# In[25]:


iris = pd.read_csv("iris.csv")


# In[26]:


iris = iris.iloc[:,1:6]


# In[27]:


iris


# In[28]:


x = iris["Sepal.Length"]


# In[29]:


y = iris["Sepal.Width"]


# In[30]:


plt.scatter(x,y)
plt.xlabel("Sepal.Length")
plt.ylabel("Petal.Length")
plt.show()


# In[31]:


from sklearn.linear_model import LinearRegression
import numpy as np


# In[32]:


x


# In[33]:


x = x.values.reshape((-1, 1))


# In[34]:


y = np.array(y)


# In[35]:


model = LinearRegression()
model.fit(x, y)
# R squared
r_sq = model.score(x, y)
r_sq


# In[36]:


beta_0 = model.intercept_
beta_1 = model.coef_[0]
print('The intercept is :', beta_0)
print('The slope is :', beta_1)


# In[37]:


#Let's creat some new columns for us to better understanding the linear regression relation
# create one extra column which is x times y.
iris["xy"] = iris["Sepal.Length"]*iris["Petal.Length"]
# create one extra column which is the square of x.
iris["$x^2$"] = iris["Sepal.Length"]**2
# create one extra column which is the square of y.
iris["$y^2$"] = iris["Petal.Length"]**2
iris


# In[38]:


# calculate beta_0 and beta_1 by using the Pearson’s Correlation Coefficient formula.
beta_0 = (sum(iris["Petal.Length"])*sum(iris["$x^2$"])-sum(iris["Sepal.Length"])*sum(iris["xy"]))/(150*sum(iris["$x^2$"]) - sum(iris["Sepal.Length"])**2)
beta_1 = (150*sum(iris["xy"]) - sum(iris["Sepal.Length"])*sum(iris["Petal.Length"]))/(150*sum(iris["$x^2$"]) - sum(iris["Sepal.Length"])**2)
print('The intercept is :', beta_0)
print('The slope is :', beta_1)


# In[39]:


# calculate beta_0 and beta_1 by using the least square method formula.
beta_1 = sum((iris["Sepal.Length"]-iris["Sepal.Length"].mean())*(iris["Petal.Length"] - iris["Petal.Length"].mean()))/sum((iris["Sepal.Length"]-iris["Sepal.Length"].mean())**2)
beta_0 = iris["Petal.Length"].mean() - beta_1*iris["Sepal.Length"].mean()
print('The intercept is :', beta_0)
print('The slope is :', beta_1)


# In[40]:


# Create Variables X and Y
x = iris["Sepal.Length"]
y = iris["Petal.Length"]
# Create the scatter plot
plt.scatter(x,y)
# draw the linear equation line
plt.plot(x, beta_1*x+beta_0)
plt.xlabel("Sepal.Length")
plt.ylabel("Petal.Length")
plt.show()


# In[ ]:





# In[ ]:




