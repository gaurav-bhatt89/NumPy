#!/usr/bin/env python
# coding: utf-8

# ## NUMPY

# ## Python - NumPy Library and Operations

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


my_list = [1,2,3]


# In[35]:


my_list


# In[29]:


type(np.array(my_list))


# In[31]:


arr = np.array(my_list)
type(arr)


# In[33]:


arr


# In[39]:


my_list2 = [[1,2,3],[4,5,6,],[7,8,9]]


# In[49]:


arr2 = np.array(my_list2)
arr2


# In[61]:


arr2.shape


# In[79]:


arr3 = (np.arange(0,9))/2


# In[81]:


arr3.reshape(3,3)


# In[89]:


np.zeros(9).reshape(3,3)


# In[91]:


np.ones(9).reshape(3,3)


# In[93]:


arr4 = ((np.ones(9))*2).reshape(3,3)


# In[95]:


arr4


# In[99]:


np.ones((5,5))*2


# In[103]:


np.linspace(0,1,9).reshape(3,3)


# In[107]:


np.eye(10)


# In[109]:


np.eye(10)*2


# In[115]:


np.random.rand(10)


# In[117]:


np.random.randn(10)


# In[123]:


np.random.randint(0,100,9).reshape(3,3)


# In[129]:


ddd = np.random.randint(0,100,9).reshape(3,3)


# In[155]:


ddd


# In[135]:


len(ddd)


# In[141]:


ddd.min()


# In[147]:


ddd.mean()


# In[151]:


ddd.argmax()


# In[153]:


ddd.max()


# In[159]:


new = np.random.randint(0,100,25).reshape(5,5)


# In[161]:


new


# In[187]:


new[:2,3:]


# In[191]:


new[1:3,1:4]


# In[193]:


new


# In[201]:


bool = new > 50


# In[209]:


bool.dtype


# In[213]:


mat1 = np.random.randint(0,100,25).reshape(5,5)
mat2 = np.random.randint(0,100,25).reshape(5,5)


# In[215]:


mat1


# In[217]:


mat2


# In[221]:


mat1 + mat2


# In[223]:


mat2 - mat1


# In[225]:


mat1 * mat2


# In[231]:


mat2 / mat1


# In[233]:


np.sqrt(mat2)


# In[237]:


np.power(mat2,2)


# #PANDAS

# In[250]:


df = pd.DataFrame(np.random.randint(0,100,25).reshape(5,5),['A','B','C','D','E'],['V','W','X','Y','Z'])
df


# In[264]:


df[['W','X']][1:3]


# In[272]:


df[['W','X']].iloc[0:-1]


# In[274]:


df


# In[276]:


df['Sum']= df.sum(axis=1)


# In[286]:


df.drop('Sum',axis=1,inplace=True)


# In[288]:


df


# In[290]:


df['Sum']=df.sum(axis=1)


# In[294]:


df.sum(axis=0)


# In[296]:


df


# In[304]:


df.loc[['A','B']]


# In[310]:


df.iloc[0:3]


# In[314]:


df > 20


# In[322]:


df[df>30].dropna()


# In[328]:


df


# In[332]:


df[df['Y']>30]


# In[340]:


df[df['Y']>30]['W'][1:]


# In[342]:


df


# In[348]:


df[(df['Y']>50)&(df['W']>51)]


# #Mising Values

# In[351]:


df


# In[357]:


new = df[df>50]


# In[359]:


new


# In[377]:


new.dropna(axis=1)


# In[381]:


new


# In[383]:


new.fillna(1)


# In[385]:


new


# In[387]:


new.fillna(df['V'].mean())


# #GroupBY

# In[390]:


dict = {'Comp':['Google','Microsoft','Google','Microsoft','Amazon',],'Person':['Rick','Nick','Dave','Jason','James'],'Sales':[200,500,600,400,700]}


# In[392]:


dict


# In[394]:


new_df = pd.DataFrame(dict)


# In[396]:


new_df


# In[428]:


new_df.groupby('Comp').sum().loc['Microsoft']


# In[432]:


new_df.groupby('Comp').count()


# In[442]:


new_df.groupby('Comp').describe().transpose()


# In[448]:


df


# In[460]:


def my_func(x):
    return x**2


# In[462]:


df.apply(my_func)


# In[466]:


df.apply(lambda x: x*2)


# In[470]:


df.describe()


# In[472]:


df


# In[474]:


df.value_counts()


# In[476]:


df.nunique()


# In[478]:


df


# In[482]:


df.sort_values('V',ascending=False)


# #Matplotlib

# In[515]:


import matplotlib.pyplot as plt


# In[517]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[519]:


x = np.linspace(0,10,10)


# In[535]:


y = aa1**3


# In[537]:


plt.plot(x,y)
plt.xlabel('Gaurav')
plt.ylabel('Bhatt')
plt.title('Love You')
plt.show()


# In[541]:


plt.subplot(1,2,1)
plt.plot(x,y)
plt.subplot(1,2,2)
plt.plot(y,x)
plt.show()


# In[553]:


fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(x,y)
plt.show()


# In[575]:


fig = plt.figure()
axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.2,0.3,0.4,0.4])
axes1.plot(x,y,'red')
axes2.plot(y,x)
plt.show()


# In[ ]:





# In[583]:


fig = plt.figure()
axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.2,0.4,0.4,0.4])
axes1.plot(x,y,'red')
axes2.plot(y,x)
plt.show()


# In[593]:


fig.savefig('name.png',dpi=200)


# In[ ]:





# In[599]:


x=  np.arange(0,10)


# In[601]:


y = x ** 2


# In[630]:


fig = plt.figure()

axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.2,0.4,0.4,0.4])

axes1.plot(x,y,'red')
axes2.plot(y,x)

axes1.set_xlabel('X axis')
axes1.set_ylabel('Y axis')
axes1.set_title('Main Chart')

axes2.set_xlabel('X axis')
axes2.set_ylabel('Y axis')
axes2.set_title('Secondary Chart')

axes1.set_xlim(0,10)
axes2.set_xlim(0,80)

plt.tight_layout()
plt.show()


# In[632]:


fig.savefig('combo_chart.png',dpi=200)


# In[646]:


fig,axes = plt.subplots(1,2)

axes[0].plot(x,y,'red')
axes[1].plot(y,x,'green')

plt.show()


# In[648]:


fig.savefig('combo_chart2.png',dpi=200)


# ## Thank You

# In[ ]:




