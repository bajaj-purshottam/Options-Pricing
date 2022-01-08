#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Fixed Parameters
r = 0.05
q = 0.01
sig = 0.5
T = 1.0


# In[3]:


# model under consideration
model = 'LogNormal'


# In[21]:


def logNormal(S,r,q,sig,SO,T):
    
    f = np.exp(-0.5*(((np.log(S/SO) - (r-q-(sig**2)/2)*T))**2))/(sig*np.sqrt(2*np.pi*T)*S)
    
    return f


# In[22]:


# Plotting lognormal density f(S/SO) for various SO
m = 1500
S = np.zeros((m,1))
for i in range(m):
    S[i] = 0.25 + i*0.25
    
f_50 = logNormal(S,r,q,sig,50,T)
f_100 = logNormal(S,r,q,sig,100,T)
f_150 = logNormal(S,r,q,sig,150,T)
f_200 = logNormal(S,r,q,sig,200,T)


# In[23]:


f_50


# In[24]:


fig = plt.figure(figsize=(10,8))
labels = []

plt.plot(S,f_50,color = 'C0')
labels.append('$S_O$=' + str(50))

plt.plot(S,f_100,color = 'C1')
labels.append('$S_O$=' + str(100))

plt.plot(S,f_150,color = 'C2')
labels.append('$S_O$=' + str(150))

plt.plot(S,f_200,color = 'C3')
labels.append('$SO_S=' + str(200))

plt.legend(labels,loc = 'upper right', ncol=2)
plt.grid(alpha=0.25)
plt.grid(True)

plt.xlabel('$S_T$')
plt.ylabel('lognormal density $f(S_T|S_O)$')
plt.savefig('logNormalVariousSpot.png')
plt.show()

