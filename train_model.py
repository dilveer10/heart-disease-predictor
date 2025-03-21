import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the processed data
processed_data = pd.read_pickle("processed_data.pkl")
X_train = processed_data["X_train"]
X_test = processed_data["X_test"]
y_train = processed_data["y_train"]
y_test = processed_data["y_test"]

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training completed and saved as 'heart_disease_model.pkl'")