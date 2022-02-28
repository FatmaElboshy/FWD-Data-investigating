#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset IMDB movies
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# We are going to inspect the dataset provided by kaggle on the TMDB website, the dataset contains a great collection of movies from 1960 to 2015 with some observational properties like runtime, budget, revenue, release year and many more.
# 
# #### Questions to be answered:
# 1. What genre is most popular over the years
# 2. Which genre is the most popular from year to year?
# 3. Top 10 directors who produced the highest number of movies over the years
# 4. What are the top 5 movie profits?
# 5. What are the least 5 movie profits?

# #### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes= True, style= 'darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Takes a dataframe category column and gets the frequency of each category
def get_cat (df):
    df=df.str.get_dummies(sep='|').sum().reset_index()
    return df


# <a id='wrangling'></a>
# ## Data Wrangling
# ### General Properties

# #### Reading the dataset

# In[3]:


tmdb=pd.read_csv('tmdb-movies.csv')
tmdb.head()


# In[4]:


tmdb['budget'].value_counts()


# In[5]:


tmdb['revenue'].value_counts()


# #### Getting information about the shape of the dataset and inspecting for null values

# In[6]:


tmdb.shape


# In[7]:


list(tmdb.columns)


# In[8]:


tmdb.info()


# ### Data Cleaning 
# #### Dropping some unnecessary Columns

# In[9]:


tmdb=tmdb.drop(columns=['homepage','imdb_id','tagline','budget_adj','revenue_adj','overview'])


# #### Filling null values with the mean for the columns of int and float datatypes

# In[10]:


tmdb.fillna(tmdb.mean(), inplace=True)
tmdb.info()


# #### Inspecting for duplicate rows

# In[11]:


sum(tmdb.duplicated())


# #### Eliminating duplicated rows

# In[12]:


tmdb.drop_duplicates(inplace=True)
sum(tmdb.duplicated())


# #### Fixing data types:
# converting 'release_date' data type from object to datetime

# In[13]:


tmdb['release_date']=pd.to_datetime(tmdb['release_date'])
tmdb.info()


# #### Checking for the number of null values in each column

# In[14]:


tmdb.isnull().sum()


# #### Separating object columns from numeric columns for ease of analysis and using 'release_year' column as a key refrence 

# In[15]:


tmdb_obj=tmdb.loc[:,['original_title','cast','director','keywords','genres','production_companies','release_year']]
tmdb_obj.head()


# #### Rechecking for the null values in the object columns dataframe

# In[16]:


tmdb_obj.isnull().sum()


# #### Eliminating null rows

# In[17]:


tmdb_obj=tmdb_obj.dropna()
tmdb_obj.isnull().sum()


# #### Checking for number of rows in the object dataframe

# In[18]:


tmdb_obj.count()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1 : What genres are the most popular over the years?

# In[19]:


tmdb_gen=tmdb_obj.copy()


# #### Getting the number of the movies produced for each genre over ther years

# In[20]:


genre=get_cat(tmdb_gen['genres'])
genre


# In[21]:


#converting the output into a dataframe for ease of use
gen=pd.DataFrame(genre)
gen.columns


# In[22]:


#sorting the output values
gen=gen[['index',0]].sort_values(0, ascending=False)


# In[23]:


#plotting genre vs count
plt.figure(figsize=(20,10))
sns.barplot(x='index', y=0, data= gen, palette='pastel')
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genre', fontsize=25)
plt.ylabel('Frequency', fontsize=25, labelpad=10)
plt.title('Most common genres over the years', fontsize=30)
plt.show()


# #### We can conclude that the most 5 common genres over the years are:
# 1. Drama
# 2. Comedy
# 3. Thriller
# 4. Action
# 5. Romance

# ### Research Question 2 : What genre is most popular from year to year?

# In[24]:


#concatinating the new categorical columns with the original dataframe 
#and droping the original 'genres' column
dummies=tmdb_gen['genres'].str.get_dummies(sep='|')
gen_df=pd.concat([tmdb_gen, dummies], axis=1).drop(["genres"], axis=1)
gen_df.info()


# In[25]:


#checking the most popular genre per year
gen_df=gen_df.groupby('release_year').sum().reset_index()
#setting the 'release_year' column as the index column
gen_df=gen_df.set_index('release_year')
gen_df.head()


# In[26]:


gen_df.idxmax(axis=1)


# In[27]:


#visualization
plt.figure(figsize=(20, 17), dpi=200)
plt.pcolor(gen_df, cmap="YlGnBu", edgecolors='white', linewidths=2) # heatmap
plt.xticks(np.arange(0.5, len(gen_df.columns), 1), gen_df.columns, fontsize=25, rotation=90)
plt.yticks(np.arange(0.5, len(gen_df.index), 1), gen_df.index, fontsize=25)
plt.title('Most popular genres from year to year', fontsize=30, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.xlabel('Genres', fontsize= 27,)
plt.ylabel('Years', fontsize=27)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=12) 
cbar.ax.minorticks_on()

plt.show()


# #### We can conclude from the heat map that old movies (60's) used to be of genre 'Drama'. With the start of a new decade (70's), a new genre joined the map which is 'Thriller', and with th start of 80's, 'Comedy' genre started to be more common as well and became even more popular than 'Thriller'.

# ### Research Question 3 : Top 10 directors who directed the highest number of movies

# In[28]:


#getting director names for each movie and separating them
dum=get_cat(tmdb['director'])


# In[29]:


#converting the output to a dataframe
dum= pd.DataFrame(dum)
dum=dum[['index',0]].sort_values(0, ascending=False).head(10)
dum


# In[30]:


#visualisation
plt.figure(figsize=(15,15))
sns.barplot(x='index', y=0, data= dum, palette='pastel')
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Directors', fontsize=25)
plt.ylabel('Number of movies', fontsize=20, labelpad=10)
plt.title('Top 10 directors with highest number of movies directed', fontsize=25)
plt.show()


# #### We conclude that the highest 3 movie production rate goes to:
# 1. Woody Allen (46 movies)
# 2. Clint Eastwood	(34 movies)
# 3. Martin Scorsese	(31 movies)

# ### Research Question 4 : What's the highest 5 movie profits?

# In[31]:


tmdb_profit= tmdb.loc[:,:]
tmdb_profit.info()


# In[32]:


#calculating profit
tmdb_profit['profit'] = tmdb_profit['revenue'] - tmdb_profit['budget']
tmdb_profit.head()


# In[33]:


#getting the max 5 profits
tmdb_profit.original_title[tmdb_profit['profit']==tmdb_profit['profit'].max()]


# #### Checking the properties of the most 5 profitable movies 

# In[34]:


#sorting values
tmdb_max=tmdb_profit.sort_values('profit', ascending=False).head(5)
tmdb_max


# In[35]:


tmdb_max['runtime'].mean()


# In[36]:


get_cat(tmdb_max['production_companies'])


# In[37]:


get_cat(tmdb_max['genres'])


# In[38]:


tmdb_max['release_year'].value_counts()


# In[39]:


tmdb_max['director'].value_counts()


# In[40]:


#visualization
plt.figure(figsize=(20,10))
sns.barplot(x='original_title', y='profit', data= tmdb_max, palette='pastel')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Movies', fontsize=25, labelpad=10)
plt.ylabel('Profit in billion', labelpad=10, fontsize=20)
plt.title('The top 5 movie profits', fontsize=25)
plt.show()


# #### We can conclude that up until 2015 'Avatar' has been the movie with the highest profits since 1960, it's profit reached around 2.5 billion USD, we can see that it was the only movie that broke the 2 billion USD record.

# ### Research Question 5 : What are the least 5 movie profits?

# In[41]:


#getting the min 5 profits
tmdb_profit.original_title[tmdb_profit['profit']==tmdb_profit['profit'].min()]


# In[42]:


#sorting values
tmdb_min=tmdb_profit[['original_title','profit','genres']].sort_values('profit', ascending=False).tail(5)
#converting getting the absolute value of the profits turning them into 'Loss'
tmdb_min['Loss']=tmdb_min['profit']*-1
tmdb_min


# In[43]:


#visualization
plt.figure(figsize=(20,10))
sns.barplot(x='original_title', y='Loss', data= tmdb_min, palette='pastel')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Movies', fontsize=25, labelpad=10)
plt.ylabel('Loss in 100 million', labelpad=10, fontsize=20)
plt.title('The top 5 movie loss', fontsize=25)
plt.show()


# #### As we can see, the least profit over the years from 1960 to 2015 has been actually the biggest loss in the history of movie production. 'The Warrior's Way' has lost about 400 million USD, taking the 1st spot, while the 2nd spot went to 'The Lone Ranger' with only about 170 million USD loss.

# ## Further inspection

# ### Runtime inspection

# In[44]:


time=tmdb.loc[:,['runtime','release_year']]
time.head()


# In[45]:


time['runtime']=time[time['runtime']>40]
time= time.groupby('release_year').mean()
time.head()


# In[46]:


#visualization
plt.figure(figsize=(20,10))
sns.barplot(x=time.index, y='runtime', data= time, palette='pastel')
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.xlabel('years', fontsize=20)
plt.ylabel('Runtime in minutes', fontsize=20, labelpad=10)
plt.title('movie runtime over the years in minutes',fontsize=25)
plt.show()


# ##### As we can see too, the average runtime has decreased over the years

# ### Vote Count inspection

# In[47]:


vote=tmdb.loc[:,['vote_count','release_year']]
vote.describe()


# In[48]:


vote= vote.groupby('release_year').sum()
vote.head()


# In[49]:


#visualization
plt.figure(figsize=(20,10))
sns.barplot(x=vote.index, y='vote_count', data= vote, palette='pastel')
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.xlabel('years', fontsize=20)
plt.ylabel('Vote count', fontsize=20, labelpad=10)
plt.title('Vote count over the years',fontsize=25)
plt.show()


# #### As we see here, the vote count has increased over the years. We can also see that 2013 had the highest overall vote counts

# ### Number of released movies per year

# In[50]:


#visualization
plt.figure(figsize=(20,10))
sns.countplot( tmdb['release_year'], palette='pastel')
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.xlabel('years', fontsize=20)
plt.ylabel('Movies released', fontsize=20, labelpad=10)
plt.title('Movies released over the years',fontsize=25)
plt.show()


# As shown above, it's obvious that 2014 has been the year with most movie releases.

# ## Conclusion

# In the first section we examined the most common Movie genres over the decades. I built my analysis based on the values of 'release_year' and the 'genres'. I could find that people have always preferred specific movie genres which are: Drama, Comedy and Thriller.
# 
# After that we analyed the most 5 movie profits, from the results we can observe that:
# 1. The most movie profits belonged to movies with average 'runtime' = 150.6 minutes
# 2. 2 of these movies were directed by 'James Cameron'.
# 3. 2 of them were produced by 'Twentieth Century Fox Film Corporation', as well as 'Lightstorm Entertainment' and 'Dentsu'.
# 4. 3 of them were released in 2015.
# 5. 4 of them were 'Action' movies

# ### Limitations:

# 1. We have used TMBD Movies dataset for our analysis and worked with popularity, revenue and runtime. Our analysis is limited to only the provided dataset.
# 2. We have no informations about the popularity and how it's measured, thus we couldn't use it in our analysis.
# 3. Half of the budget and revenue values are zeros which affected the statical analysis making the distribution skewed 
# 4. There are a lot of outliers in budget ,revenue and even more outliers in budget_adj and revenue_adj
# 5. The dataset information are provided till 2015 only and isn't updated.
