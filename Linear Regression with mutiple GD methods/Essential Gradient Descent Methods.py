
# coding: utf-8

# ## Part.1 - Batch Gradient Descent

# In[7]:


import pandas as pd
import numpy as np
train = pd.read_csv('C:/Users/tianjiayang/mike_is_building_wheels/Linear Regression with mutiple GD methods/data/train.csv')
test = pd.read_csv('C:/Users/tianjiayang/mike_is_building_wheels/Linear Regression with mutiple GD methods/data/train.csv')


# In[3]:


train.head()


# ### Step 1. Initialization

# In[39]:



# Initialize parmeters
beta = [1, 1]
#learning rate
alpha = 0.2
# tolerence for stop
tol_L = 0.01

# feature_scalin
max_x = max(train['id'])
x = train['id'] / max_x
y = train['questions']

# An helper function to help us compute rmse
def rmse(beta, x, y):
    squared_err = (beta[0] + beta[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res


# ### Step 2. Compute gradient

# In[40]:


# a function to compute current gradient of the objective function
def compute_grad(beta, x, y):
    grad = [0, 0]
    #where x and y is the whole data set
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
    grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))
    return np.array(grad)


# ### Step 3. Update

# In[41]:



def update_beta(beta, alpha, grad):
    new_beta = np.array(beta) - alpha * grad
    return new_beta


# ### Step 4. Iterating...

# In[42]:



# First Iteration
loss = rmse(beta, x, y)
grad = compute_grad(beta, x, y)
beta = update_beta(beta, alpha, grad)
loss_new = rmse(beta, x, y)

# Iterating...
i = 1
# Stop criterion:
while np.abs(loss_new - loss) > tol_L:
    
    beta = update_beta(beta, alpha, grad)
    grad = compute_grad(beta, x, y)
    loss = loss_new
    loss_new = rmse(beta, x, y)
    i += 1
    print('Round %s Diff RMSE %s'%(i, abs(loss_new - loss)))
print('Coef: %s \nIntercept %s'%(beta[1], beta[0]))


# We've been throught 204 rounds iteration and finally reached criterion
# 
# Because we've normalized x, so we need to get it back to get real coefficents.
# 

# In[43]:


print('Our Coef: %s \nOur Intercept %s'%(beta[1] / max_x, beta[0]))


# In[44]:


res = rmse(beta, x, y)
print('Our RMSE: %s'%res)


# Comparing with scikit-learn's LinearRegression

# In[45]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[['id']], train[['questions']])
print('Sklearn Coef: %s'%lr.coef_[0][0])
print('Sklearn Coef: %s'%lr.intercept_[0])


# In[46]:


res = rmse([936.051219649, 2.19487084], train['id'], y)
print('Sklearn RMSE: %s'%res)


# We got pretty close coefficients and RMSE

# ## Part2. Stochastic Gradient Descent

# In[50]:



import pandas as pd
import numpy as np
train = pd.read_csv('C:/Users/tianjiayang/mike_is_building_wheels/Linear Regression with mutiple GD methods/data/train.csv')
test = pd.read_csv('C:/Users/tianjiayang/mike_is_building_wheels/Linear Regression with mutiple GD methods/data/train.csv')

# Initialization
beta = [1, 1]
alpha = 0.2
tol_L = 0.001
max_x = max(train['id'])
x = train['id'] / max_x
y = train['questions']



#  Instead of use whole dataset, we randomly choose a point of the dataset to optimize our objective function
def compute_grad_SGD(beta, x, y):
    grad = [0, 0]
    #randomly choose a point
    r = np.random.randint(0, len(x))
    # update gradient based on that point
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
#     grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
#     grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))
    return np.array(grad)

# Same as before
def update_beta(beta, alpha, grad):
    new_beta = np.array(beta) - alpha * grad
    return new_beta

# Same as before
def rmse(beta, x, y):
    squared_err = (beta[0] + beta[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res

# First iteration
np.random.seed(101)
grad = compute_grad_SGD(beta, x, y)
loss = rmse(beta, x, y)
beta = update_beta(beta, alpha, grad)
loss_new = rmse(beta, x, y)

# Iterating
i = 1
while np.abs(loss_new - loss) > tol_L:
    beta = update_beta(beta, alpha, grad)
    grad = compute_grad_SGD(beta, x, y)
#     if i % 100 == 0:
    loss = loss_new
    loss_new = rmse(beta, x, y)
    print('Round %s Diff RMSE %s'%(i, abs(loss_new - loss)))
    i += 1

print('Coef: %s \nIntercept %s'%(beta[1], beta[0]))


# In[51]:


print('Our Coef: %s \nOur Intercept %s'%(beta[1] / max_x, beta[0]))


# In[53]:


res = rmse(beta, x, y)
print('Our RMSE: %s'%res)


# ### Part3. Mini-batch Gradient Descent
# 
# An trade-off between full-batch and stochastic

# We can set the batch size, which means we use how many data points in one iteration
# 
# When b = 1, Mini-batch = SGD, and when b = n, mini-batch = full-batch GD
# 

# In[55]:



import pandas as pd
import numpy as np


train = pd.read_csv('C:/Users/tianjiayang/mike_is_building_wheels/Linear Regression with mutiple GD methods/data/train.csv')
test = pd.read_csv('C:/Users/tianjiayang/mike_is_building_wheels/Linear Regression with mutiple GD methods/data/train.csv')


beta = [1, 1]
alpha = 0.2
tol_L = 0.001
# set the batch size
batch_size = 16


max_x = max(train['id'])
x = train['id'] / max_x
y = train['questions']


def compute_grad_batch(beta, batch_size, x, y):
    grad = [0, 0]
    # randomly choose 16 data points
    r = np.random.choice(range(len(x)), batch_size, replace=False)
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)


def update_beta(beta, alpha, grad):
    new_beta = np.array(beta) - alpha * grad
    return new_beta


def rmse(beta, x, y):
    squared_err = (beta[0] + beta[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res


np.random.seed(10)
grad = compute_grad_batch(beta, batch_size, x, y)
loss = rmse(beta, x, y)
beta = update_beta(beta, alpha, grad)
loss_new = rmse(beta, x, y)


i = 1
while np.abs(loss_new - loss) > tol_L:
    beta = update_beta(beta, alpha, grad)
    grad = compute_grad_batch(beta, batch_size, x, y)
#     if i % 100 == 0:
    loss = loss_new
    loss_new = rmse(beta, x, y)
    print('Round %s Diff RMSE %s'%(i, abs(loss_new - loss)))
    i += 1
print('Coef: %s \nIntercept %s'%(beta[1], beta[0]))


# In[56]:


print('Our Coef: %s \nOur Intercept %s'%(beta[1] / max_x, beta[0]))


# In[57]:


res = rmse(beta, x, y)
print('Our RMSE: %s'%res)

