### INF601 - Advanced Programming in Python
### Samuel Amoateng
### Mini Project 2


# ### Used Price Car Prediction 
# 
# **Question:  "What factors most significantly impact used car prices?"**

# In[269]:

import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
import random 

random.seed(42)
pd.set_option("display.max_columns", None)
# [x for x in dir(random) if x.startswith("s")]


# In[270]:


dataset = pd.read_csv("Used_Car_Price_Prediction.csv")
dataset.head()


# In[271]:


dataset.info()


# In[272]:


dataset.describe()


# #### Now we deal with the null values in the dataset to make it balanced 

# In[273]:


dataset.isnull().sum()


# **Here we are going to check for the null or missing values and deal with them separately**

# In[274]:


dataset.isnull().sum()[dataset.isnull().sum() > 0]


# In[275]:


null_ = dataset.isnull().sum()[dataset.isnull().sum() > 0].index 
null_features = list(null_)


# In[276]:


null_features


# In[277]:


null_data = dataset[null_features].copy()


# In[278]:


null_data.head()


# In[279]:


# pd.set_option("display.max_colwidth", None)
null_data.describe(include="all") # We now describe all the values to see how we will take care of the null data


# In[280]:


for i in null_features:
    print(f"{i}:{null_data[i].unique()}")


# #### We now fill the data with the missing values 

# In[281]:


dataset['body_type'] = dataset['body_type'].fillna("Unknown")
dataset['transmission'] = dataset['transmission'].fillna("Unknown")
dataset['registered_city'] = dataset['registered_city'].fillna("Unknown")
dataset['registered_state'] = dataset['registered_state'].fillna("Unknown")
dataset['source'] = dataset['source'].fillna("Unknown")
dataset['car_availability'] = dataset['car_availability'].fillna("Unknown")  # thease are nominal data


dataset['car_rating'] = dataset['car_rating'].fillna(dataset['car_rating'].mode()[0])
dataset['fitness_certificate'] = dataset['fitness_certificate'].fillna(dataset['fitness_certificate'].mode()[0]) # These are ordinal and boolean 

dataset['original_price'] = dataset['original_price'].fillna(dataset['original_price'].median()) # This is numeric data

# We are filling missing values by replacing them. 


# In[282]:


dataset.info()


# ##### All null values have been assigned and given values 

# ### Making all features numeric 

# In[283]:


dataset.head()


# In[284]:


dataset.describe(include = "all")


# In[285]:


dataset.dtypes


# In[286]:


object_data = []
for i in dataset:
    if dataset[i].dtype.name == "object":
        object_data.append(i)
object_data    # Categorical features having more then two classes 


# In[ ]:





# In[287]:


for i in object_data:
    print(f"{i}:{dataset[i].unique()}")


# In[288]:


bool_data = []
for i in dataset:
    if dataset[i].dtype.name == "bool":
        bool_data.append(i)
bool_data  # this is the categorical features for boolean 


# In[289]:


int_data = []
for i in dataset:
    if dataset[i].dtype.name == "int64" or dataset[i].dtype.name == "float64":
        int_data.append(i)
int_data # Numerical features 


# ### Correlation analysis

# In[290]:


copy_data = dataset.copy()
target_data = copy_data["sale_price"]
copy_data.drop("sale_price",axis=1)
copy_data.drop("booking_down_pymnt",axis=1) 
copy_data.drop("emi_starts_from",axis=1)
copy_data.head()


# In[291]:


num_corr_data= copy_data[int_data].corr(method = "pearson")
num_corr_data["sale_price"].sort_values(ascending = False)


# In[292]:


plt.figure(figsize=(10,10))
sns.heatmap(num_corr_data, xticklabels = num_corr_data.columns, yticklabels = num_corr_data.columns , annot = True)
plt.show()


# In[293]:


print(bool_data)
print(object_data)


# In[294]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_correlation = ['assured_buy', 'is_hot', 'fitness_certificate', 'reserved', 'warranty_avail', 'car_availability', 'car_rating', 'ad_created_on']

encode = OrdinalEncoder()
copy_data[ordinal_correlation] = encode.fit_transform(copy_data[ordinal_correlation])


# In[295]:


copy_data.info()


# In[296]:


ord_corr_data= copy_data[ordinal_correlation + ["sale_price"]].corr(method = "spearman")
ord_corr_data["sale_price"].sort_values(ascending = False)


# In[297]:


plt.figure(figsize=(10,10))
sns.heatmap(ord_corr_data, xticklabels = ord_corr_data.columns, yticklabels = ord_corr_data.columns , annot = True)
plt.show()


# In[298]:


import os

# Create charts folder
if not os.path.exists("charts"):
    os.makedirs("charts")

top_ordinal = ['car_rating', 'fitness_certificate', 'ad_created_on']

for col in top_ordinal:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=copy_data[col], y=target_data)
    plt.title(f"Sale Price vs {col}")
    plt.xlabel(col)
    plt.ylabel("Sale Price")
    # Save figure
    plt.savefig(f"charts/boxplot_{col}.png", bbox_inches="tight")
    plt.close()


# In[299]:


top_numerical = ['broker_quote', 'emi_starts_from', 'original_price']

for col in top_numerical:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=copy_data[col], y=target_data)
    sns.regplot(x=copy_data[col], y=target_data, scatter=False, color="red")  # optional regression line
    plt.title(f"Sale Price vs {col}")
    plt.xlabel(col)
    plt.ylabel("Sale Price")
    # Save figure
    plt.savefig(f"charts/scatter_{col}.png", bbox_inches="tight")
    plt.close()


# In[300]:


# Example for ordinal
ord_corr_vals = ord_corr_data['sale_price'].drop('sale_price').sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=ord_corr_vals.values, y=ord_corr_vals.index, palette="viridis")
plt.title("Ordinal Feature Correlation with Sale Price")
plt.xlabel("Spearman Correlation")
plt.ylabel("Feature")
plt.savefig("charts/ordinal_corr_barplot.png", bbox_inches="tight")
plt.show()
plt.close()


# In[303]:


num_corr_vals = num_corr_data["sale_price"].drop(labels=["sale_price", "emi_starts_from" , "booking_down_pymnt"])
num_corr_vals = num_corr_vals.sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=num_corr_vals.values, y=num_corr_vals.index, palette="magma")
plt.title("Numerical Feature Correlation with Sale Price")
plt.xlabel("Pearson Correlation")
plt.ylabel("Feature")
plt.show()
plt.savefig("charts/numerical_corr_barplot.png", bbox_inches="tight")
plt.close()


# In[ ]:






# This project will be using Pandas dataframes. This isn't intended to be full blown data science project. The goal here is to come up with some question and then see what API or datasets you can use to get the information needed to answer that question. This will get you familar with working with datasets and asking questions, researching APIs and gathering datasets. If you get stuck here, please email me!
# (5/5 points) Initial comments with your name, class and project at the top of your .py file.
# (5/5 points) Proper import of packages used.
# (20/20 points) Using a data source of your choice, such as data from data.gov or using the Faker package, generate or retrieve some data for creating basic statistics on. This will generally come in as json data, etc.
# Think of some question you would like to solve such as:
# "How many homes in the US have access to 100Mbps Internet or more?"
# "How many movies that Ridley Scott directed is on Netflix?" - https://www.kaggle.com/datasets/shivamb/netflix-shows
# Here are some other great datasets: https://www.kaggle.com/datasets
# (10/10 points) Store this information in Pandas dataframe. These should be 2D data as a dataframe, meaning the data is labeled tabular data.
# (10/10 points) Using matplotlib, graph this data in a way that will visually represent the data. Really try to build some fancy charts here as it will greatly help you in future homework assignments and in the final project.
# (10/10 points) Save these graphs in a folder called charts as PNG files. Do not upload these to your project folder, the project should save these when it executes. You may want to add this folder to your .gitignore file.
# (10/10 points) There should be a minimum of 5 commits on your project, be sure to commit often!
# (10/10 points) I will be checking out the main branch of your project. Please be sure to include a requirements.txt file which contains all the packages that need installed. You can create this fille with the output of pip freeze at the terminal prompt.
# (20/20 points) There should be a README.md file in your project that explains what your project is, how to install the pip requirements, and how to execute the program. Please use the GitHub flavor of Markdown. Be thorough on the explanations.