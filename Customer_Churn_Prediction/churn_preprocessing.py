import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


data = pd.read_csv(r'Data\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna()

# Encode categorical features
le = LabelEncoder()
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
for feature in categorical_features:
    data[feature] = le.fit_transform(data[feature])

# Feature and target split
X = data.drop(columns=['customerID', 'Churn'])
y = le.fit_transform(data['Churn'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize numeric features only
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])


X_train.to_csv('Data/churn_X_train.csv', index=False)
X_test.to_csv('Data/churn_X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('Data/churn_y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('Data/churn_y_test.csv', index=False)
