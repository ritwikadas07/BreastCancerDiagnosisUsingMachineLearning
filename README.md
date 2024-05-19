# Breast Cancer Classification Using Machine Learning

## Overview

This repository contains a robust machine learning approach for accurately classifying breast tumors as benign or malignant. Using the Breast Cancer Wisconsin (Diagnostic) dataset, which is derived from fine-needle aspiration (FNA) images, we compare multiple machine learning algorithms to determine the most accurate classification model.

## Abstract

Breast cancer remains a critical global health challenge, highlighting the importance of early detection. This study leverages machine learning algorithms to classify breast tumors using features extracted from FNA images. The algorithms evaluated include logistic regression, decision trees, random forest, k-nearest neighbors (KNN), support vector machine (SVM), and Gaussian Naive Bayes. The aim is to develop a reliable model for early diagnosis, aiding healthcare professionals in optimizing treatment strategies.

## Keywords

- Breast Cancer
- Support Vector Machine (SVM)
- Predictive Modeling
- Healthcare Diagnostic
- Tumor Classification
- Logistic Regression
- Decision Trees
- Random Forest
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Correlation Analysis
- Multicollinearity
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Cross-Validation

## Workflow

### Dataset and Features

The Breast Cancer Wisconsin (Diagnostic) dataset includes ten real-valued attributes extracted from FNA images of cell nuclei:
- radius_mean
- texture_mean
- perimeter_mean
- area_mean
- smoothness_mean
- compactness_mean
- concavity_mean
- concave_points_mean
- symmetry_mean
- fractal_dimension_mean

### Data Cleaning

Data cleaning involves:
- Removing irrelevant columns (e.g., ID).
- Encoding categorical values 'M' (Malignant) and 'B' (Benign) into numerical labels (1 and 0, respectively).

<p align="center">
<img src="/images/Picture1.png" "Data Cleaning removing irrelevant columns and encoding categorical values">
</p>

### Feature Selection

Correlation analysis is performed to identify and remove highly correlated predictors to address multicollinearity. Features with strong correlations are dropped to improve model performance and interpretability.

### Preprocessing

Data is standardized using techniques like Standard Scaler to ensure each feature contributes equally to the model. Exploratory data analysis (EDA) is conducted to summarize essential aspects of the dataset.

### Model Training and Evaluation

The dataset is split into training (70%) and testing (30%) sets. Various machine learning models are implemented and trained, including:
- Logistic Regression
- Decision Trees
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

### Evaluation Metrics

Models are evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Cross-validation (e.g., tenfold cross-validation) is used to ensure model robustness.

## Outputs

The support vector machine (SVM) emerged as the top-performing model with an accuracy of 96.4%. It demonstrated superior performance in handling high-dimensional data effectively.

### Evaluation Results

1. **Logistic Regression**
   - Accuracy: 95%
   - Confusion Matrix: 
     - True Positive: 110
     - False Positive: 5
     - False Negative: 2
     - True Negative: 54

<p align="center">
<img src="/images/Picture2.png" "Confusion matrix and Accuracy for Logistic Regression">
</p>

2. **Decision Tree**
   - Accuracy: 91%
   - Confusion Matrix:
     - True Positive: 105
     - False Positive: 10
     - False Negative: 5
     - True Negative: 51

<p align="center">
<img src="/images/Picture3.png" "Confusion matrix and Accuracy for Decision Tree">
</p>

3. **Random Forest**
   - Accuracy: 92%
   - Confusion Matrix:
     - True Positive: 108
     - False Positive: 7
     - False Negative: 5
     - True Negative: 51

<p align="center">
<img src="/images/Picture4.png" "Confusion matrix and Accuracy for Random Forest">
</p>

4. **Support Vector Machine (SVM)**
   - Accuracy: 96.4%
   - Confusion Matrix:
     - True Positive: 112
     - False Positive: 3
     - False Negative: 4
     - True Negative: 48

<p align="center">
<img src="/images/Picture5.png" "Confusion matrix and Accuracy for Support Vector Machine">
</p>

## Conclusion

The SVM classifier stands out with its exceptional accuracy in distinguishing between benign and malignant breast tumors. This model provides healthcare professionals with a reliable tool for early diagnosis and effective treatment planning, ultimately enhancing patient outcomes.

## Future Work

Future research could explore deep learning approaches, such as convolutional neural networks (CNNs), and advanced feature engineering techniques to further improve predictive accuracy.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The authors would like to thank the UCI Machine Learning Repository for providing the Breast Cancer Wisconsin (Diagnostic) dataset.
- Special thanks to healthcare professionals and researchers for their contributions to breast cancer diagnosis and treatment.
