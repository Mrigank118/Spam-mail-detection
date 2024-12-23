# Spam Detection using Machine Learning

This project implements a machine learning pipeline to classify emails as spam or not spam using a dataset with word frequencies, character frequencies, and other email features. The model is trained using logistic regression, and its performance is evaluated using various metrics and visualizations.

---

## Features

- **Data Exploration**: Visualize the dataset to understand the distribution of features.
- **Feature Importance**: Identify the most influential features for classification.
- **Model Training**: Train a logistic regression model on the dataset.
- **Performance Metrics**: Evaluate the model using accuracy, confusion matrix, ROC curve, and more.
- **Visualizations**: Graphical representation of feature distributions and model performance.

---

## Dataset Description

The dataset contains the following features:

- `word_freq_*`: Frequency of specific words in the email.
- `char_freq_*`: Frequency of specific characters in the email.
- `capital_run_length_*`: Metrics related to sequences of capital letters.
- `class`: The target variable (1 = Spam, 0 = Not Spam).

---

## Requirements

- Python 3.8+
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `seaborn`
  - `matplotlib`

Install dependencies using:
```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

---

## Project Workflow

1. **Data Preprocessing**:
   - Load the dataset using `pandas`.
   - Split the dataset into training and testing sets.
   - Standardize the feature values.

2. **Model Training**:
   - Train a logistic regression model on the training set.

3. **Evaluation**:
   - Calculate accuracy and visualize the confusion matrix.
   - Plot the ROC curve to evaluate model performance.

4. **Feature Importance Analysis**:
   - Extract and visualize the most important features based on model coefficients.

5. **Feature Distributions**:
   - Visualize the distributions of the top features using KDE plots and histograms.

---

## Key Visualizations

1. **Confusion Matrix**:
   - Understand the model's classification performance.

   ![Confusion Matrix](images/confusion_matrix.png)

2. **ROC Curve**:
   - Evaluate the trade-off between sensitivity and specificity.

   ![ROC Curve](images/roc_curve.png)

3. **Top Features**:
   - Visualize the distributions of the top features.

   ![Feature Distributions](images/feature_distributions.png)

---

## Example Code

### Model Training and Evaluation
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Navigate to the project directory:
   ```bash
   cd spam-detection
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook or Python script:
   ```bash
   jupyter notebook
   ```

---

## Future Improvements

- Implement additional classifiers (e.g., SVM, Random Forest).
- Experiment with feature engineering techniques.
- Use cross-validation for more robust evaluation.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- Dataset inspired by email spam classification tasks.
- Visualization techniques supported by `seaborn` and `matplotlib`.
