import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


X_train = pd.read_csv('Data/churn_X_train.csv')
X_test = pd.read_csv('Data/churn_X_test.csv')
y_train = pd.read_csv('Data/churn_y_train.csv').values.ravel()
y_test = pd.read_csv('Data/churn_y_test.csv').values.ravel()


model = GradientBoostingClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Calculate accuracy on training set
train_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy}")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, 'churn_detection_model.pkl')