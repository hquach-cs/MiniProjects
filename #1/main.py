# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import matplotlib.pyplot as plt

GDP = pd.read_csv("GDP.csv",thousands=',')
LE = pd.read_csv("LE.csv")
Data = {'Country':GDP['Country Name'],'GDP':GDP['2017'],'LE':LE['2017']}
Data = pd.DataFrame(Data,columns=['Country','GDP','LE'])
Data = Data.dropna()
x = Data['LE'].to_numpy()
y = Data['GDP'].to_numpy()
plt.scatter(x,y)


# %%
import sklearn 
import numpy as np

model = sklearn.linear_model.LinearRegression()
x = np.reshape(x,(-1,1))
model.fit(x,y)
plt.scatter(x,y)
plt.plot(x,model.predict(x),color='k')


# %%


