# 🏠 House Price Prediction using Multiple Linear Regression (From Scratch)

## 📖 Overview

This project implements **Multiple Linear Regression from scratch using NumPy** to predict house prices based on multiple housing features. Instead of relying on machine learning libraries such as **scikit-learn**, every component of the algorithm—including prediction, cost function, gradient computation, and gradient descent—has been implemented manually to understand the mathematical foundations of Linear Regression.

This project follows the complete Machine Learning workflow from data analysis to model evaluation.

---

## 🎯 Project Objectives

- Perform Exploratory Data Analysis (EDA)
- Preprocess and prepare the dataset
- Implement Multiple Linear Regression from scratch
- Train the model using Gradient Descent
- Predict house prices
- Evaluate model performance
- Visualize model results

---

## 📂 Dataset

**Dataset:** Housing.csv

**Target Variable:**
- Price

**Features Used:**
- Area
- Bedrooms
- Bathrooms
- Stories
- Main Road
- Guest Room
- Basement
- Hot Water Heating
- Air Conditioning
- Parking
- Preferred Area
- Furnishing Status (One-Hot Encoded)

---

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📊 Exploratory Data Analysis

Performed the following analyses:

- Dataset Information
- Missing Value Analysis
- Statistical Summary
- Target Variable Analysis
- Numerical Feature Analysis
- Categorical Feature Analysis
- Correlation Heatmap
- Scatter Plots
- Distribution Plots
- Outlier Detection using Boxplots

---

## ⚙️ Data Preprocessing

- Checked Missing Values
- One-Hot Encoding
- Feature Selection
- Train-Test Split
- Feature Scaling (Standardization)

---

## 🧠 Model Implementation

The complete Multiple Linear Regression algorithm was implemented manually using NumPy.

Implemented components include:

- Prediction Function
- Cost Function
- Gradient Function
- Gradient Descent
- Model Training
- House Price Prediction

**No machine learning libraries (e.g., scikit-learn) were used for building the model.**

---

## 📈 Model Performance

| Metric | Value |
|---------|-------|
| **MAE** | ₹779,710.27 |
| **MSE** | 1.146 × 10¹² |
| **RMSE** | ₹1,070,742.18 |
| **R² Score** | **0.6609** |

The model explains approximately **66% of the variation** in house prices.

---

## 📉 Visualizations

The project includes:

- Cost vs Iterations
- Actual vs Predicted House Prices
- Residual Plot
- Prediction Error Distribution
- Correlation Heatmap
- Feature Distribution Plots
- Boxplots

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── Housing.csv
├── multiple_regression.ipynb
├── README.md
├── requirements.txt
└── graphs/
    ├── cost_vs_iterations.png
    ├── actual_vs_predicted.png
    ├── residual_plot.png
    └── prediction_error_distribution.png
    
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/Sahilbhayre/House-Price-Prediction.git
```

Move into the project folder

```bash
cd House-Price-Prediction
```

Install the required packages

```bash
pip install -r requirement.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open **multiple_regression.ipynb** and run all cells.

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Multiple Linear Regression Mathematics
- Cost Function
- Gradient Computation
- Gradient Descent Optimization
- Model Evaluation
- Regression Visualization

---

## 🚀 Future Improvements

- Linear Regression using Scikit-Learn
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Hyperparameter Tuning
- Streamlit Web App Deployment

---

## 👨‍💻 Author

**Sahil Bhayre**

- GitHub: https://github.com/SahilBhayre
- Portfolio: https://sahilbhayre.github.io/My-Portfolio/
- LinkedIn: https://www.linkedin.com/in/sahil-bhayre-b2029b238/

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!
