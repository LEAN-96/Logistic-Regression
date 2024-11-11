# Logistic Regression

## 1. Project Overview

This project demonstrates the implementation of **Logistic Regression**, a supervised learning algorithm used to classify data into binary categories—in this case, whether a user will click on an ad or not. It works by estimating the probability that a given input belongs to a specific class (click vs. no-click) using a logistic (sigmoid) function, which maps the output to values between 0 and 1.

The notebook walks through the entire process, from loading and preprocessing data, to building and training a logistic regression model, and finally evaluating its performance using various metrics.

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
   - Model performance is evaluated using metrics such as accuracy score, precision, recall, and F1-score.
   
5. **Visualization**:
   - Users can expect visualizations like accuracy plots over training iterations to better understand model performance.

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

Once MyBinder loads:
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

After running the logistic regression model, you will see a classification report that includes several key metrics used to evaluate the model's performance: precision, recall, f1-score, and support. Each metric is calculated for both classes (those that clicked on the ad and those that did not).

- **Accuracy**:
  - Indicates the overall proportion of correct predictions. Here, the model correctly classifies 97% of instances.

- **Precision**:
  - Measures the accuracy of positive predictions (clicks). For users who clicked, a precision of 0.98 means that 98% of predictions are correct.
  
- **Recall**:
  - Measures the ability to capture actual positives (clicks). A recall of 0.96 for users who clicked indicates that the model correctly identifies 96% of actual clicks.
  
- **F1-Score**:
  - The harmonic mean of precision and recall, providing a single measure of balance. An F1-score of 0.97 for clicked ads shows strong model performance.

- **Support**:
  - The number of occurrences of each class (clicked and not clicked) in the test data.

### Summary
In short, logistic regression helps to predict the probability of a user clicking on an ad based on their characteristics, allowing for informed decision-making and efficient ad targeting.
The model shows robust performance across all metrics, effectively distinguishing between users who clicked on the ad and those who didn’t.

By following these steps, users can fully replicate all experiments conducted in this machine learning project.