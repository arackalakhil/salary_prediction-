# 💼 Salary Prediction Project

A simple machine learning project using **Linear Regression** to predict employee salaries based on various features like age, gender, education level, job title, and years of experience.

---

## 📌 Overview

This project uses a Jupyter Notebook (`first_prediction_project.ipynb`) to:

- Explore and preprocess salary data
- Train a Linear Regression model
- Visualize actual vs. predicted salaries
- Evaluate the model’s performance

---

## 📊 Dataset

**File:** `Salary Data.csv`

**Columns:**

- `Age`: Age of the employee *(numeric)*
- `Gender`: Gender *(Male/Female)*
- `Education Level`: Bachelor’s, Master’s, or PhD *(categorical)*
- `Job Title`: Job role *(categorical)*
- `Years of Experience`: Total experience *(numeric)*
- `Salary`: Annual salary *(target)*

**Notes:**

- Total records: 375
- Missing values: 2 rows (must be handled)
- Outliers: Minimum salary of $350 may need review

---

## 📁 Project Structure

salary-prediction-project/
│
├── first_prediction_project.ipynb # Main notebook
├── Salary Data.csv # Dataset (not included)
├── README.md # This file
└── requirements.txt # Dependencies
🛠️ Notebook Workflow
1. Data Loading & Exploration
Load the CSV file

Check for missing values and outliers

View dataset structure and stats

2. Data Preprocessing (to be completed)
Drop missing values

Encode categorical features with one-hot encoding

Scale numerical columns with StandardScaler

3. Model Training (to be completed)
Split data into train/test

Train a LinearRegression model

4. Prediction & Visualization
Predict test set salaries

Visualize 20 actual vs. predicted salaries with bar chart

📊 Example Output
Index	Actual	Predicted
0	180,000.00	171,912.18
1	65,000.00	103,906.32
2	125,000.00	141,242.01
3	80,000.00	74,586.75
4	140,000.00	142,592.61
