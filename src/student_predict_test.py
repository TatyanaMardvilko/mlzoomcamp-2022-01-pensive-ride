#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests


# In[2]:


url = "http://localhost:9696/predict"


# In[3]:


student = {
"gender": "female",
"race_ethnicity": "group_C",
"parental_level_of_education": "bachelor's_degree",
"lunch": "standard",
"test_preparation_course": "completed"
}


# In[4]:


student


# In[6]:


response = requests.post(url, json=student).json()
response


# In[ ]:




