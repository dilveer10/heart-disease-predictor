import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
file_path = "heart.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Convert categorical variables to numeric using Label Encoding
label_encoder = LabelEncoder()
df["Sex"] = label_encoder.fit_transform(df["Sex"])
df["ChestPainType"] = label_encoder.fit_transform(df["ChestPainType"])
df["RestingECG"] = label_encoder.fit_transform(df["RestingECG"])
df["ExerciseAngina"] = label_encoder.fit_transform(df["ExerciseAngina"])
df["ST_Slope"] = label_encoder.fit_transform(df["ST_Slope"])

# Separate features (X) and target variable (y)
X = df.drop(columns=["HeartDisease"])  # Features
y = df["HeartDisease"]  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the numerical data (improves model performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the processed data
processed_data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
}
pd.to_pickle(processed_data, "processed_data.pkl")

print("Data Preprocessing Completed!")