# Heart Failure Detection System - ML Dashboard

A comprehensive machine learning web application for heart failure risk prediction, built with React, TypeScript, and modern ML algorithms.

## 🎯 Features

- **CSV Upload & Auto-Detection**: Upload cardiovascular datasets with automatic schema detection
- **Data Quality Analysis**: Comprehensive data quality reports with missing value detection, duplicate identification, and column type classification
- **Exploratory Data Analysis**: Interactive visualizations including distributions, correlations, and target class analysis
- **Multiple ML Models**: Train and compare 5 different classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors
- **Model Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score, specificity, ROC-AUC, and confusion matrices
- **Prediction System**: Make predictions with trained models and download results
- **Feature Importance**: Analyze which features contribute most to predictions

## 🚀 Quick Start

### Using Sample Data
1. Click "Upload Dataset" in the sidebar
2. Click "Load Sample Data" to load a pre-generated dataset (299 patients, 12 features)
3. The system will automatically detect the target column (DEATH_EVENT)
4. Click "Validate & Clean" to prepare the data
5. Click "Train Models" to train all 5 classifiers
6. View results in the Evaluation page
7. Make predictions in the Prediction page

### Using Your Own Data
1. Prepare a CSV file with cardiovascular data
2. Ensure it has a target column (binary or categorical)
3. Upload via "Upload Dataset"
4. Select the target column if not auto-detected
5. Follow the same workflow as above

## 📊 Expected Dataset Format

Your CSV should contain patient/cardiovascular data with features and a target column. Example formats:

**UCI Heart Disease Dataset format:**
```
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
```

**Heart Failure Clinical Records format:**
```
age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, smoking, time, DEATH_EVENT
```

### Auto-Detected Target Column Names
The system automatically detects target columns named:
- DEATH_EVENT, HeartDisease, heart_failure, target, label, outcome, diagnosis, class, result, status

## 🔧 Workflow

### 1. Upload Dataset
- Drag and drop a CSV file or click to browse
- Or use the sample data for testing
- System displays dataset overview (rows, columns, missing values)

### 2. Data Quality Review
- Review column types and statistics
- Select or confirm the target column
- Click "Validate & Clean" to:
  - Remove duplicates
  - Handle missing values (median imputation for numerical, mode for categorical)
  - Exclude identifier columns
  - Prepare data for training

### 3. Exploratory Data Analysis
- View target class distribution
- Analyze missing values by column
- Explore feature correlations (heatmap)
- Check feature distributions (histograms)
- Review descriptive statistics

### 4. Model Training
- Click "Train Models" on the Data Quality page or Training page
- System automatically:
  - Performs stratified train/test split (80/20)
  - Scales features (standardization)
  - Trains 5 different models
  - Evaluates on test set

### 5. Model Evaluation
- Compare all models side-by-side
- View detailed metrics table
- Analyze confusion matrices
- Check ROC curves
- Identify the best model (based on F1-score)
- Download evaluation report as CSV

### 6. Make Predictions
- Enter feature values manually or use "Fill Sample Values"
- Get prediction from the best model
- View predicted class and probability
- Download prediction results as CSV

### 7. Feature Importance
- Select a model to view feature importance
- Compare importance across all models
- Understand which features drive predictions

## 📈 Model Performance Metrics

- **Accuracy**: Overall correctness (TP + TN) / Total
- **Precision**: Positive predictive value TP / (TP + FP)
- **Recall (Sensitivity)**: True positive rate TP / (TP + FN)
- **Specificity**: True negative rate TN / (TN + FP)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## ⚠️ Important Disclaimers

### Educational Use Only
This application is designed for **educational and research purposes only**. It is **NOT** a clinically validated diagnostic tool.

### Not Medical Advice
- Predictions are model-estimated probabilities, not confirmed medical risks
- Do not use this system for actual medical decision-making
- Always consult qualified healthcare professionals for medical concerns
- Model performance does not guarantee clinical effectiveness

### Limitations
- Models are trained on uploaded data only
- No external validation or clinical testing
- Results may not generalize to different populations
- Feature importance shows statistical associations, not causation

## 🛠️ Technical Details

### Data Processing
- **Missing Values**: Median imputation (numerical), mode imputation (categorical)
- **Duplicates**: Removed automatically
- **Feature Scaling**: Standardization (zero mean, unit variance)
- **Train/Test Split**: Stratified 80/20 split to preserve class distribution
- **Random Seed**: 42 for reproducibility

### Machine Learning
- **Logistic Regression**: L2 regularization, learning rate 0.01, 500 epochs
- **Decision Tree**: Gini impurity, max depth 8, min samples 5
- **Random Forest**: 30 trees, bootstrap sampling, max depth 6
- **Gradient Boosting**: 50 estimators, learning rate 0.1, max depth 3
- **KNN**: K=5, Euclidean distance

### Technologies
- React 18 with TypeScript
- Tailwind CSS for styling
- Recharts for visualizations
- PapaParse for CSV parsing
- Custom ML implementations (no external ML libraries)

## 🐛 Troubleshooting

### "No cleaned dataset available"
- Solution: Click "Validate & Clean" on the Data Quality page first

### "No target column selected"
- Solution: Select a target column from the dropdown on the Data Quality page

### "Model training failed"
- Check browser console for error details
- Ensure dataset has at least 2 classes in the target column
- Verify there are enough samples (minimum 50 recommended)

### "Prediction failed"
- Ensure all required fields are filled
- Check that input values are numeric
- Verify models have been trained first

### Models not showing in Evaluation
- Wait for training to complete (check loading indicator)
- Check browser console for errors
- Ensure dataset passed validation

## 📝 Example Workflow with Sample Data

1. **Upload**: Click "Load Sample Data"
2. **Auto-detection**: System detects DEATH_EVENT as target
3. **Validation**: Click "Validate & Clean"
   - Removes 0 duplicates
   - Imputes 0 missing values
   - Identifies 11 feature columns
4. **Training**: Click "Train Models"
   - Trains 239 samples, tests on 60 samples
   - Trains 5 models in ~2 seconds
5. **Evaluation**: View results
   - Best model typically: Random Forest or Gradient Boosting
   - Accuracy: ~75-85%
   - ROC-AUC: ~0.75-0.85
6. **Prediction**: Enter values or use sample
   - Get prediction and probability
   - Download results

## 🔬 Research Notes

### Class Imbalance
The system detects and warns about class imbalance. If imbalance ratio > 3:1:
- Consider using class weights (not yet implemented)
- Be cautious about accuracy as a metric
- Focus on recall for medical screening applications

### Small Datasets
For datasets < 50 samples:
- Results may be unreliable
- Cross-validation not yet implemented
- Consider collecting more data

### Feature Engineering
Current implementation:
- Automatic one-hot encoding for categorical features
- No manual feature engineering yet
- No feature selection (all features used)

## 📄 License

This project is for educational purposes. Use responsibly and ethically.

## 🤝 Contributing

This is a demonstration project. For production use, consider:
- Adding cross-validation
- Implementing class weight handling
- Adding more models (SVM, Neural Networks)
- Implementing feature selection
- Adding model persistence (save/load)
- Implementing proper train/validation/test splits

## 📞 Support

For issues or questions:
- Check browser console for error messages
- Review the troubleshooting section
- Verify dataset format matches expected structure

---

**Remember**: This is an educational tool. Always consult healthcare professionals for medical decisions.
#   H e a r t - f a i l u r e - d e t e c t i o n - s y s t e m  
 #   H e a r t - f a i l u r e - d e t e c t i o n - s y s t e m  
 