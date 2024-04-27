#!/usr/bin/env python
# coding: utf-8

# # Week-1
# 
# # Linear Regression Example using tensorflow
# Linear regression implementation with TensorFlow v2 library.
# 
# This example is using a low-level approach to better understand all mechanics behind the training process.
# 

# ![image.png](attachment:image.png)

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
rng = np.random


# In[1]:





# ![image.png](attachment:image.png)

# In[ ]:





# In[2]:


learning_rate = 0.01
training_step = 1000
display_step = 50


# In[ ]:


# Just run the next cell using 'Ctrl' + 'Enter'


# In[3]:


# Training Data.

X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])

Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = X.shape[0]


# ![image.png](attachment:image.png)

# In[4]:


print(X, '\n', X.shape)


# In[4]:





# ![image.png](attachment:image.png)

# In[5]:


print(Y, '\n', Y.shape)


# In[5]:





# ![image.png](attachment:image.png)

# In[19]:


W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

print(w.numpy(),'\n', b.numpy())


# In[6]:





# ![image.png](attachment:image.png)

# In[20]:


def linear_regression(x):
    return W * x + b


# In[7]:





# ![image.png](attachment:image.png)

# In[27]:


def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred-y_true, 2)) / (2 * n_samples)


# In[8]:





# ![image.png](attachment:image.png)

# In[28]:


optimizer = tf.optimizers.SGD(learning_rate)


# In[10]:





# In[ ]:


# Adding a question mark () after the function name lets you view the latest documentation. 
# This is a handy tool.


# ![image.png](attachment:image.png)

# In[29]:


get_ipython().run_line_magic('pinfo', 'tf.optimizers.SGD')


# In[12]:





# ![image.png](attachment:image.png)

# In[30]:


def run_optimization():
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        
    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


# In[13]:





# ![image.png](attachment:image.png)

# In[31]:


for step in range(1, training_step + 1):
    run_optimization()
    
    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))


# In[14]:





# ![image.png](attachment:image.png)

# In[32]:


plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, np.array(W*X+b), label='Fitted line')
plt.legend()
plt.show()








