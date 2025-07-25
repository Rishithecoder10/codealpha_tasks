# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset from a public source
# This dataset contains information about used cars.
url = 'https://raw.githubusercontent.com/driveu/data-science-project/main/car%20data.csv'
car_data = pd.read_csv(url)

print("First 5 rows of the car dataset:")
print(car_data.head())

# Drop the 'Car_Name' column as it's not a useful feature for a general model
car_data = car_data.drop('Car_Name', axis=1)

# Separate target variable (y) and features (X)
X = car_data.drop('Selling_Price', axis=1)
y = car_data['Selling_Price']

# Identify categorical and numerical features
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
numerical_features = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']

# Create a preprocessing pipeline
# This will apply one-hot encoding to categorical features and leave numerical features as is.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create the regression model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', LinearRegression())])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Mean Squared Error: {mse:.2f}")
print(f"Model R-squared: {r2:.2f}")

# Example of predicting the price of a new car
# Create a sample DataFrame with the same feature columns as the training data
new_car = pd.DataFrame({
    'Year': [2018],
    'Present_Price': [10.0],
    'Kms_Driven': [50000],
    'Fuel_Type': ['Petrol'],
    'Seller_Type': ['Dealer'],
    'Transmission': ['Manual'],
    'Owner': [0]
})

predicted_price = model.predict(new_car)
print(f"\nPredicted price for the new car: {predicted_price[0]:.2f} lakhs")
