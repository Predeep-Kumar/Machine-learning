import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('Data/creditcard.csv')

# Feature and target split
X = data.drop(columns=['Class'])
y = data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pd.DataFrame(X_train).to_csv('Data/X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('Data/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('Data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('Data/y_test.csv', index=False)
