import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib


# Load the dataset
df = pd.read_csv('./data/penguins.csv')

imputer = SimpleImputer(strategy='mean')  # Use mean strategy for imputation
#Fit the imputer on the dataset to calculate the imputation values for each feature
imputer.fit(df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']])
#Apply the imputer to fill in the missing values in the specified features:
df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']] = imputer.transform(df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']])

# Remove rows with missing values if any (only sex variable should be missing at this point thanks to imputer)
df.dropna(inplace=True)

#convert sex feature to integer
lb = LabelEncoder()
df["sex"] = lb.fit_transform(df["sex"])

# Select relevant features and target variable
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
target = 'species'

# Encode the target variable
df[target] = lb.fit_transform(df[target])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Perform cross-validation (5 fold)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# calculate f1 score (weighted considers class imbalance)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy}")
print(f"f1 score: {f1}")

# Example prediction on new data
new_data = pd.DataFrame([[39.2, 18.5, 197.0, 4200, 1]], columns=features)
predicted_species = lb.inverse_transform(model.predict(new_data))
print(f"Predicted species: {predicted_species}")

#save the trained model
joblib.dump(model, "./model/trained_model.pkl")

