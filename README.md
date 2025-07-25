# CodeAlpha_Data_Science_Projects

This repository contains the projects completed during my data science internship at CodeAlpha.The internship provided hands-on experience with data preprocessing, building predictive models, and deploying machine learning solutions[cite: 7].

As per the internship instructions, I have completed two tasks from the Data Science track. The source code for each task is included in this repository.

---

## Task 1: Iris Flower Classification 🌸

### Project Objective
The goal of this project is to build a machine learning model that can accurately classify the species of an Iris flower—**setosa**, **versicolor**, or **virginica**—based on its sepal and petal measurements. [cite_start]This is a classic project for understanding fundamental classification concepts in machine learning[cite: 30].

### Methodology
* **Dataset**: The project uses the well-known Iris dataset, which is easily accessible through the `scikit-learn` library.
* **Model**: A **K-Nearest Neighbors (KNN)** classifier was trained to learn the relationship between the flower's features and its species.
* **Libraries Used**: `pandas`, `scikit-learn`.
* **Evaluation**: The model's performance was evaluated on a test set, and it demonstrated high accuracy in classifying the flowers.

### How to Run the Code
1.  Ensure you have Python and the required libraries (`pandas`, `scikit-learn`) installed.
2.  Run the Python script for this task to see the model training process and the final classification report.

---

## Task 2: Car Price Prediction 🚗

### Project Objective
This project aims to predict the selling price of used cars by training a regression model. The model uses various features of a car to estimate its value, demonstrating a practical application of machine learning in price prediction.

### Methodology
* [cite_start]**Dataset**: The dataset includes features like the car's original price (`Present_Price`), manufacturing year (`Year`), kilometers driven (`Kms_Driven`), and `Fuel_Type`.
* **Model**: A **Linear Regression** model was trained to predict the continuous value of the car's selling price.
* **Libraries Used**: `pandas`, `scikit-learn`.
* [cite_start]**Feature Engineering**: The project involved preprocessing the data, including using one-hot encoding for categorical features like `Fuel_Type`, `Seller_Type`, and `Transmission`[cite: 42].
* **Evaluation**: The model was evaluated using metrics like Mean Squared Error (MSE) and R-squared to assess its prediction accuracy.

### How to Run the Code
1.  Make sure Python and the necessary libraries (`pandas`, `scikit-learn`) are installed.
2.  Execute the Python script to see how the model is trained and used to predict car prices.
