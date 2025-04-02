# K-Nearest Neighbors (KNN) Classifier - Gene Expression Dataset

## Overview
This project implements a **K-Nearest Neighbors (KNN) classifier** to predict whether cancer is present based on gene expression levels. The dataset contains gene expression values for two genes, and the target variable indicates whether the patient has cancer.

## Dataset
- The dataset used is **gene_expression.csv**.
- It consists of the following columns:
  - `Gene One`: Expression level of the first gene.
  - `Gene Two`: Expression level of the second gene.
  - `Cancer Present`: Target variable (1 = Cancer present, 0 = No cancer).

## Implementation Steps
1. **Data Loading & Exploration**
   - Read the dataset using pandas.
   - Check for missing values and data types.
   - Perform summary statistics and visualize distributions.
   
2. **Data Preprocessing & Visualization**
   - Box plots and distribution plots for each gene.
   - Scatter plot to observe gene correlation with cancer presence.
   - Check for skewness and correlations.

3. **Feature Engineering & Splitting**
   - Separate features (`Gene One`, `Gene Two`) and target variable (`Cancer Present`).
   - Split data into training (70%) and testing (30%) sets.
   - Standardize the features using `StandardScaler`.

4. **Model Training & Evaluation**
   - Train a **KNN classifier** with default parameters.
   - Predict on training and test sets.
   - Evaluate accuracy and perform **cross-validation**.

5. **Hyperparameter Tuning**
   - Find the optimal `k` value by testing values from 1 to 30.
   - Use `GridSearchCV` to optimize `k` and distance metric (`p`).

6. **Final Model Evaluation**
   - Train the best model with optimized `k`.
   - Evaluate performance using:
     - **Confusion matrix**
     - **Classification report** (Precision, Recall, F1-score)
     - **Accuracy score**

## Installation & Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/KNN-GeneExpression.git
   cd KNN-GeneExpression
   ```
2. Run the Jupyter Notebook or Python script.
   ```bash
   jupyter notebook KNN_Intuition.ipynb
   ```
3. Modify parameters and visualize results.

## Results
- The best `k` value was found to be **20**.
- The model achieved **high accuracy** on test data.
- The confusion matrix and classification report showed good classification performance.

## Future Improvements
- Experiment with different feature scaling techniques.
- Try **other classification algorithms** like Decision Trees or SVM.
- Use more features for better accuracy.

## Contributing
Feel free to fork and improve this project!

## License
This project is open-source under the MIT License.

