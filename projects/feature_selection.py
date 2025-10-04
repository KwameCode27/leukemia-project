import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#loading dataset
data = pd.read_csv('merged.csv' , encoding_errors= 'replace')

#create scaled DataFrame where each variable has mean of 0 and standard dev of 1
scaler = StandardScaler()
scaled_df = scaler.fit_transform(data)
scaled_df = data
#plt.plot(scaled_df) plt.show()
plt.figure(figsize=(12,12))
cor1 = data.corr()
sns.heatmap(cor1, annot=True, cmap="coolwarm",annot_kws={"size":8})
plt.show() 

# As I said above the data can be divided into three parts.lets divied the features according to their category
features_mean=(scaled_df.columns[1:22])
# lets check the correlation between features
# now we will do this analysis only for features_mean then we will do for others and will see who is doing best
corr = data[features_mean].corr() # .corr is used for find corelation
plt.figure(figsize=(12,12))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'coolwarm') 
plt.show()