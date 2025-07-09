import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load the dataset from the file
data = pd.read_csv('dataset.csv', header=None, names=['domain', 'is_dga']) # dataset.csv has to be replaced with the respective dataset file name

# Step 2: Feature extraction function
def extract_features(domain):
    return {
        'length': len(domain),
        'num_digits': sum(c.isdigit() for c in domain),
        'num_special_chars': sum(not c.isalnum() for c in domain),
        'num_vowels': sum(c in 'aeiou' for c in domain)
    }

# Step 3: Extract features for each domain in the dataset
features = data['domain'].apply(extract_features).tolist()
features_df = pd.DataFrame(features)

# Step 4: Prepare the dataset
X = features_df  # Feature set
y = data['is_dga']  # Target variable

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 8: Function to classify a new domain
def classify_domain(domain):
    features = extract_features(domain)  # Extract features for the input domain
    features_df = pd.DataFrame([features])  # Create a DataFrame for prediction
    prediction = model.predict(features_df)  # Make prediction
    return "DGA" if prediction[0] == 1 else "Legitimate"

# Example usage
new_domain = input("Enter a domain to classify: ")  # Get input from user
classification = classify_domain(new_domain)  # Classify the domain
print(f"The domain '{new_domain}' is classified as: {classification}")