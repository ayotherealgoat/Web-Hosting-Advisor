import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Step 1: Load the dataset
df = pd.read_csv('hosting_providers.csv')
df.columns = df.columns.str.strip()

# Convert unlimited values to large numbers and handle new tech stacks
def convert_resource(value):
    if isinstance(value, str) and value.lower() == 'unlimited':
        return 999999
    try:
        return float(value)
    except:
        return value

# Convert unlimited to numeric values
df['bandwidth'] = df['bandwidth'].apply(convert_resource)
df['storage'] = df['storage'].apply(convert_resource)

# Step 2: Define features and target
X = df[['cost', 'uptime', 'storage', 'bandwidth', 'tech_stack', 'control_panel']]

# Update tech stack mapping
numeric_mappings = {
    'tech_stack': {
        'PHP': 1.0, 'Python': 1.0, 'Node.js': 1.0, 
        'Ruby': 1.0, 'Windows': 0.0, 'Custom': 1.0
    },
    'control_panel': {
        'cPanel': 1.0, 'hPanel': 0.0, 'Custom': 0.0, 
        'vDeck': 0.0, 'SPanel': 1.0
    }
}

for col, mapping in numeric_mappings.items():
    X[col] = X[col].map(mapping)

X = X.astype('float64')  # Convert all to float64
y = df['recommended'].astype('int64')

# Step 3: Preprocessing for categorical features
categorical_features = ['tech_stack', 'control_panel']
numerical_features = ['cost', 'uptime', 'storage', 'bandwidth']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')

# Step 4: Build pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Fit model
model.fit(X_train, y_train)

# Step 7: Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl.")
