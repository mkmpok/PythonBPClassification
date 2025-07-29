import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load the dataset
df = pd.read_csv('patient_data.csv')

# Mapping the Stages column to numeric labels (if not already done)
if 'Stages_Label' not in df.columns:
    stage_mapping = {
        'NORMAL': 0,
        'HYPERTENSION (Stage-1)': 1,
        'HYPERTENSION (Stage-2)': 2
    }
    df['Stages_Label'] = df['Stages'].map(stage_mapping)

# Define features to label encode
label_cols = ['Age', 'C', 'History', 'Patient', 'TakeMedication', 'Severity',
              'BreathShortness', 'VisualChanges', 'NoseBleeding', 'Whendiagnoused',
              'Systolic', 'Diastolic', 'ControlledDiet']

# Encode categorical features
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Handle missing/invalid types

# Rename column 'C' to 'gender'
df.rename(columns={'C': 'gender'}, inplace=True)

# Drop rows with any missing or infinite values
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.dropna(inplace=True)

# Prepare features and target
X = df.drop(['Stages_Label', 'Stages'], axis=1, errors='ignore')
y = df['Stages_Label']

# Save model input column names
model_columns = X.columns.tolist()
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train and save the model
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Cleaned data, trained model, and saved model + scaler + columns successfully.")
print(df.isnull().sum())
