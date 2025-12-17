## Project Title

Fake Alcohol Detection Using Machine Learning

## Project Overview

Counterfeit and adulterated alcohol poses serious public health and safety risks due to the presence of toxic substances such as methanol. This project implements a supervised machine learning–based system to classify alcohol samples as **genuine** or **fake** using chemical composition data. The solution combines data preprocessing, feature selection, multiple classification models, cross-validation, hyperparameter tuning, and a Flask-based web dashboard for visualization.

The dataset used contains approximately 7,000 records with numerical chemical attributes such as acidity levels, sulphates, alcohol percentage, methanol content, and impurity concentration. The final system evaluates multiple models and identifies the most reliable classifier based on performance metrics.

---

## Key Features

* Binary classification of alcohol samples (Genuine vs Fake)
* Implementation of multiple ML models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting
  * K-Nearest Neighbors (KNN)
  * Support Vector Machine (RBF)
* Handling overfitting using attrition testing, feature selection, and cross-validation
* Hyperparameter tuning using GridSearchCV
* Performance evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix
* Visual analytics including feature importance, KDE plots, scatter plots, and comparison graphs
* Flask-based web dashboard for result visualization

---

## Project Structure

```
fake-alcohol-detection/
│── app.py
│── fake_alcohol_dataset.csv
│── best_model.pkl
│── static/
│   └── plots/
│       ├── *_confusion.png
│       ├── FeatureImportance.png
│       ├── PerformanceComparison.png
│── templates/
│   ├── index.html
│   └── results.html
│── README.md
```

---

## Tools & Technologies

* Programming Language: Python 3.9+
* Libraries:

  * pandas, numpy
  * scikit-learn
  * matplotlib, seaborn
  * flask
  * joblib

---

## How to Run the Project

1. Install Python (version 3.9 or higher)
2. Install required libraries:

   ```
   pip install pandas numpy scikit-learn matplotlib seaborn flask joblib
   ```
3. Place `fake_alcohol_dataset.csv` in the project directory
4. Run the Flask application:

   ```
   python app.py
   ```
5. Open a web browser and navigate to:

   ```
   http://127.0.0.1:5000/
   ```

---

## Output

* Model comparison table showing performance metrics
* Confusion matrices for each model
* Feature importance plot (Random Forest)
* Feature distribution and scatter plots
* Best-performing model highlighted and saved as `best_model.pkl`

---

## Contributors

* Seher Sanghani – Data preprocessing, visualization, Flask implementation
* Om Sankar Nadar – Exploratory data analysis, model training
* Sara Deshmukh – Model training, attrition testing
* Antriksha Jain – Model training, attrition testing
