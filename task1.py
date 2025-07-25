# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to a pandas DataFrame for better visualization (optional)
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

print("First 5 rows of the dataset:")
print(df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Display a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Example of predicting a new flower
# Measurements: sepal length, sepal width, petal length, petal width
new_flower = [[5.1, 3.5, 1.4, 0.2]] # Example for setosa
prediction = knn.predict(new_flower)
predicted_species = iris.target_names[prediction[0]]
print(f"\nPrediction for new flower with measurements {new_flower[0]}: {predicted_species}")
