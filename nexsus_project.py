#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max.columns",None)
import pandasql as sql


# In[2]:


disease = pd.read_csv(r"C:\Users\mothe\Desktop\nexus\The_Cancer_data_1500_V2.csv", header=0)
disease_BK = disease.copy()
disease.copy()


# In[3]:


disease.info()


# In[4]:


disease.isnull().sum()


# In[5]:


disease_dup = disease[disease.duplicated(keep='last')]
disease_dup


# In[6]:


disease.columns


# In[7]:


cols=['Age','BMI','Smoking','GeneticRisk','PhysicalActivity','AlcoholIntake']


# In[8]:


IndepVar = []
for col in disease.columns:
    if col != 'Diagnosis':
        IndepVar.append(col)

TargetVar = 'Diagnosis'

x = disease[IndepVar]
y = disease[TargetVar]


# In[9]:


from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Display the shape for train & test data

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[10]:


from sklearn.preprocessing import MinMaxScaler    # to normalise the data
mmscaler=MinMaxScaler(feature_range=(0,1))
x_train=mmscaler.fit_transform(x_train)   # overall taking on x_train 
x_train=pd.DataFrame(x_train)
x_test=mmscaler.fit_transform(x_test)
x_test=pd.DataFrame(x_test)


# In[18]:


# Build the multi regression model

from sklearn.linear_model import LinearRegression  

# Create object for the model

ModelMLR = LinearRegression()
#ModelMLR = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None, positive=False)                   

# Train the model with training data

ModelMLR.fit(x_train, y_train)

# Predict the model with test dataset

y_pred = ModelMLR.predict(x_test)

# Evaluation metrics for Regression analysis

from sklearn import metrics

print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, y_pred),3))  
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, y_pred),3))  
print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),3))
print('R2_score:', round(metrics.r2_score(y_test, y_pred),6))
#print('Root Mean Squared Log Error (RMSLE):', round(np.log(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),3))

# Define the function to calculate the MAPE - Mean Absolute Percentage Error

def MAPE (y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Evaluation of MAPE 

result = MAPE(y_test, y_pred)
print('Mean Absolute Percentage Error (MAPE):', round(result, 3), '%')

# Calculate Adjusted R squared values 

r_squared = round(metrics.r2_score(y_test, y_pred),6)
adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),6)
print('Adj R Square: ', adjusted_r_squared)


# In[19]:


# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



# Step 5: Train the Support Vector Machine model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(x_train, y_train)

# Step 6: Make predictions on the test set
y_pred = svm_model.predict(x_test)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
file_path =r"C:\Users\mothe\Downloads\The_Cancer_data_1500_V2.csv" 
disease = pd.read_csv(file_path, header=0)

# Assuming the target column is named 'target' and the rest are features
# Adjust this based on your dataset's structure
X = disease.drop(columns='Diagnosis')
y = disease['Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the SVM model
svm_model = SVC(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Implement cross-validation and hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, 
                           cv=5, scoring='accuracy', n_jobs=-1)

# Train the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the final SVM model with the best parameters
best_svm_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_svm_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-validation scores for the best model
cv_scores = cross_val_score(best_svm_model, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




