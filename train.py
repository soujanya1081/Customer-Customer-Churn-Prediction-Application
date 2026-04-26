import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset
df = pd.read_csv('telecom_churn_data.csv')

# 2. Data Cleaning
# TotalCharges has some empty strings; convert to numeric and drop those rows
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges', 'MonthlyCharges', 'gender'], inplace=True)

# 3. Feature Selection
# We use the key features requested for the app
features = ['gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 
            'Contract', 'PaymentMethod', 'InternetService', 'TechSupport', 'OnlineSecurity']
target = 'Churn'

X = df[features]
y = df[target]

# 4. Encoding Categorical Variables
# We will use a simple LabelEncoder for each object column
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Encode Target (No=0, Yes=1)
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 7. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Save Model and Scaler
pickle.dump(model, open('churn_model.pkl', 'wb'))
pickle.dump(scaler, open('churn_scaler.pkl', 'wb'))

print("Success: 'churn_model.pkl' and 'churn_scaler.pkl' created!")