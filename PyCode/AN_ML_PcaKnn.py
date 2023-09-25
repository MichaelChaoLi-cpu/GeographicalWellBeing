# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 10:59:52 2023

@author: Li Chao
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

used_columns = ['LIFE_SAT', 'HAPPY',
                'MENTAL_HEALTH', 'PHYSICAL_HLTH',
                'WORTHWHILE', 'LIFE_PURPOSE',
                'PROMOTE_GOOD', 'GIVE_UP',
                'CONTENT', 'SAT_RELATNSHP',
                'EXPENSES', 'WORRY_SAFETY']
data_sweden = pd.read_csv('../Data/gfs_sweden_w1_perturbed_sample.csv')
summary = data_sweden[used_columns].describe()

data_sweden[used_columns] = data_sweden[used_columns].replace(-98, np.nan)
summary = data_sweden[used_columns].describe()

### replace -98 with mean values of columns
column_means = data_sweden[used_columns].mean()
data_sweden[used_columns] = data_sweden[used_columns].fillna(column_means)
summary = data_sweden[used_columns].describe()

#### to six domains
pca = PCA(n_components=1)
### Happiness and Life Satisfaction
data = data_sweden[['LIFE_SAT', 'HAPPY']]
principal_component = pca.fit_transform(data)
data_sweden['HaLS'] = principal_component
information_retained = pca.explained_variance_ratio_.sum()
print(f"Information Retained: {information_retained * 100:.2f}%")
HaLS_IR = information_retained
# Information Retained: 71.35%

### Mental and Physical Health
data = data_sweden[['MENTAL_HEALTH', 'PHYSICAL_HLTH']]
principal_component = pca.fit_transform(data)
data_sweden['MaPH'] = principal_component
information_retained = pca.explained_variance_ratio_.sum()
print(f"Information Retained: {information_retained * 100:.2f}%")
MaPH_IR = information_retained
# Information Retained: 62.16%

### Meaning and Purpose
data = data_sweden[['WORTHWHILE', 'LIFE_PURPOSE']]
principal_component = pca.fit_transform(data)
data_sweden['MaPu'] = principal_component
information_retained = pca.explained_variance_ratio_.sum()
print(f"Information Retained: {information_retained * 100:.2f}%")
MaPu_IR = information_retained
# Information Retained: 68.35%

### Character and Virtue
data = data_sweden[['PROMOTE_GOOD', 'GIVE_UP']]
principal_component = pca.fit_transform(data)
data_sweden['CaV'] = principal_component
information_retained = pca.explained_variance_ratio_.sum()
print(f"Information Retained: {information_retained * 100:.2f}%")
CaV_IR = information_retained
# Information Retained: 59.46%

### Close Social Relationships
data = data_sweden[['CONTENT', 'SAT_RELATNSHP']]
principal_component = pca.fit_transform(data)
data_sweden['CSR'] = principal_component
information_retained = pca.explained_variance_ratio_.sum()
print(f"Information Retained: {information_retained * 100:.2f}%")
CSR_IR = information_retained
# Information Retained: 73.80%

### Financial and Material Stability
data = data_sweden[['EXPENSES', 'WORRY_SAFETY']]
principal_component = pca.fit_transform(data)
data_sweden['FaMS'] = principal_component
information_retained = pca.explained_variance_ratio_.sum()
print(f"Information Retained: {information_retained * 100:.2f}%")
FaMS_IR = information_retained
# Information Retained: 74.21%

information_retained = [HaLS_IR, MaPH_IR, MaPu_IR, CaV_IR, CSR_IR, FaMS_IR]

#### to one level
data = data_sweden[['HaLS', 'MaPH', 'MaPu', 'CaV', 'CSR', 'FaMS']]
principal_component = pca.fit_transform(data)
data_sweden['Flourish_Value'] = principal_component
information_retained = pca.explained_variance_ratio_.sum()
print(f"Information Retained: {information_retained * 100:.2f}%")
# Information Retained: 45.01%

Flourish_Value_median = data_sweden['Flourish_Value'].median()
data_sweden['Flourish_Value_Binary'] = (data_sweden['Flourish_Value'] >= Flourish_Value_median).astype(int)
##### ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##### This method is retired:
#####   this method is investigate who get most variation.    

#### KNN test
X = data_sweden[used_columns]
y = data_sweden['Flourish_Value_Binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=42)
knn = KNeighborsClassifier(n_neighbors=2)  
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# Accuracy: 0.9024552090245521

#### KMeans 
data = data_sweden[used_columns]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_scaled)
clusters = kmeans.labels_
data_sweden['Flourish_Value_Binary_kmeans'] = clusters
data_sweden['Flourish_Value_Binary_kmeans'].describe()

#### KNN test
X = data_sweden[used_columns]
y = data_sweden['Flourish_Value_Binary_kmeans']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=42)
knn = KNeighborsClassifier(n_neighbors=2)  
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# Accuracy: 0.9283344392833444


data_sweden.to_csv('../Data/gfs_sweden_w1_perturbed_sample_Classified.csv')

col_of_interest = used_columns + ['Flourish_Value_Binary_kmeans']
filtered_df = data_sweden.loc[data_sweden['Flourish_Value_Binary_kmeans'] == 1, col_of_interest]
description = filtered_df.describe()
print(description)

filtered_df = data_sweden.loc[data_sweden['Flourish_Value_Binary_kmeans'] == 0, col_of_interest]
description_neg = filtered_df.describe()
print(description_neg)

description.loc['mean',:] - description_neg.loc['mean',:]





