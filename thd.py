import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
input_file = pd.read_csv('LoanApprovalPrediction.csv')
input_file.drop(columns=['Loan_ID'], axis=1, inplace=True)

# Impute missing values for numeric columns
numeric_cols_to_impute = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in numeric_cols_to_impute:
    input_file[col] = SimpleImputer(strategy='median').fit_transform(input_file[[col]])

# Impute missing values for categorical columns
categorical_cols_to_impute = ['Dependents', 'Education', 'Self_Employed', 'Property_Area']
for col in categorical_cols_to_impute:
    input_file[col] = SimpleImputer(strategy='most_frequent').fit_transform(input_file[[col]])

# Label Encoding for binary categorical variables
label_encoder = LabelEncoder()
binary_categorical_cols = ['Gender', 'Married']
for col in binary_categorical_cols:
    input_file[col] = label_encoder.fit_transform(input_file[col])

# Encode categorical variables (all other categorical columns)
categorical_cols = ['Dependents', 'Education', 'Self_Employed', 'Property_Area']
input_file = pd.get_dummies(input_file, columns=categorical_cols, drop_first=True)

# Data Splitting
X = input_file.drop(columns=['Loan_Status'], axis=1)
y = input_file['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=99)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)

# Predicting the test set results
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Display confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', confusion)

# Display classification report
classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)

# Plot a histogram of predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]
plt.hist(y_prob, bins=20)
plt.xlabel('Predicted Probabilities')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities')
plt.show()
