# Logistic Regression

## 1. Project Overview

This project demonstrates the implementation of **Logistic Regression**, a widely-used machine learning algorithm for binary classification problems. The notebook walks through the entire process, from loading and preprocessing data, to building and training a logistic regression model, and finally evaluating its performance using various metrics. Users can expect to learn how to apply logistic regression to a dataset, interpret results such as accuracy and confusion matrices, and visualize model performance.

### Dataset
The dataset used in this notebook is a synthetic dataset generated for binary classification purposes. It contains features representing independent variables and a target variable indicating the class label (0 or 1). The dataset is preprocessed by standardizing the features to ensure that all variables contribute equally to the logistic regression model.

### Machine Learning Methods

- **Logistic Regression**: Logistic regression is a linear model used for binary classification tasks. It estimates the probability that an instance belongs to a particular class using a logistic function. In this notebook, logistic regression is applied to predict binary outcomes based on input features.

### Notebook Overview

1. **Data Loading and Preprocessing**:
   - The dataset is loaded into the notebook using pandas. Features are standardized using `StandardScaler` from scikit-learn to ensure that all input variables are on the same scale.
   
2. **Model Building**:
   - A **Logistic Regression** model is defined using scikit-learn's `LogisticRegression` class. The model is set up with default parameters, but users can modify regularization strength and solver options as needed.
   
3. **Model Training**:
   - The logistic regression model is trained on the preprocessed data using the `fit()` method. The training process involves minimizing the log-loss function to find optimal model parameters.
   
4. **Evaluation**:
   - Model performance is evaluated using metrics such as accuracy score, precision, recall, and F1-score. A **confusion matrix** is also generated to visualize the number of true positives, true negatives, false positives, and false negatives.
   
5. **Visualization**:
   - Users can expect visualizations like accuracy plots over training iterations and confusion matrix heatmaps to better understand model performance.

## 2. Requirements

### Running Locally

To run this project locally on your machine:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/LEAN-96/Logistic-Regression.git
    cd logistic-regression
    ```

2. **Set up a virtual environment**:

    Using `venv`:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

    Or using `conda`:
    ```bash
    conda create --name ml-env python=3.8
    conda activate ml-env
    ```

3. **Install project dependencies**:
    Install all required Python packages by running:
    ```bash
    pip install -r requirements.txt
    ```

4. **Launch Jupyter Notebook**:
    Start Jupyter Notebook by running:
    ```bash
    jupyter notebook
    ```
    Open the notebook (`Logistic_Regression.ipynb`) in your browser through the Jupyter interface.

### Running Online via MyBinder

For users who prefer not to install any software locally, you can run this notebook online using MyBinder.

Click the MyBinder button below to launch the notebook in an interactive environment:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LEAN-96/Logistic-Regression.git/HEAD?labpath=notebooks)

nce MyBinder loads:
1. Navigate to your notebook (`Logistic_Regression.ipynb`) in the file browser on the left.
2. Click on the notebook file to open it.
3. Run all cells by selecting "Run All" from the "Cell" menu or pressing `Shift + Enter` for individual cells.

By using MyBinder, you can explore and execute all parts of this notebook without installing anything locally.

## 3. Reproducing Results

To reproduce the results shown in this project:

1. Open the notebook (`Logistic_Regression.ipynb`) either locally or online via MyBinder.
2. Execute all cells sequentially by selecting them and pressing `Shift + Enter`.
3. Ensure that all cells execute without errors.
4. Observe output results such as accuracy metrics and visualizations directly in the notebook.

### Interpreting Results:

- **Accuracy Metrics**: These metrics show how well the logistic regression model performs on both training and test datasets.
- **Confusion Matrix**: This matrix helps visualize correct vs incorrect predictions across different classes (0 or 1).
- **Feature Analysis or Graphs**: Visual representations such as decision boundaries or feature importance may be included depending on how the notebook is structured.

By following these steps, users can fully replicate all experiments conducted in this machine learning project.