# Iris Classification: From Binary to Multiclass with Scikit-Learn

This project explores binary and multiclass classification using the classic **Iris dataset**. The notebook covers data preprocessing, visualization, model building, evaluation, and hyperparameter tuning using various machine learning algorithms from **scikit-learn**.

## Project Highlights

* **Dataset**: Built-in `sklearn.datasets.load_iris()`
* **Tech stack**: Python, NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib
* **Models**: Logistic Regression, Decision Tree, Support Vector Machine (SVM)
* **Concepts Covered**:

  * Data exploration & correlation analysis
  * Binary classification (subset of Iris dataset)
  * Multiclass classification with logistic regression, decision trees, and SVM
  * Hyperparameter tuning with `GridSearchCV`
  * Pipelines and scaling with `StandardScaler`
  * Decision tree structure inspection
  * SVM decision boundary visualization (2D)

## Setup

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/iris-ml-classification.git
cd iris-ml-classification
pip install -r requirements.txt
```

**Required packages:**

* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `seaborn`

Alternatively, run the notebook directly on [Kaggle](https://www.kaggle.com/) or in a Jupyter environment.

## Notebook Overview

### 1. Binary Classification

* Filtered classes for a binary task
* Exploratory data analysis (EDA)
* Logistic Regression with and without feature scaling
* Evaluation using accuracy and F1 score
* Hyperparameter tuning via `GridSearchCV`

### 2. Multiclass Classification

* Visualized pairwise feature relationships
* Compared Logistic Regression, Decision Tree, and SVM
* Tuned hyperparameters for each model
* Visualized decision boundaries for SVM (2D)

## Example Outputs

* Decision tree plot via `sklearn.tree.plot_tree`
* Feature correlations and scatter plots with Seaborn
* Text-based tree structure using `tree_` attributes
* SVM decision surface via `DecisionBoundaryDisplay`

## File Structure

```
📁 iris-ml-classification/
│
├── iris_classification.ipynb     # Main notebook
├── README.md                     # Project overview
├── requirements.txt              # Python dependencies
```

## Future Improvements

* Add cross-validation performance visualizations
* Introduce ensemble methods (e.g., Random Forest, Gradient Boosting)
* Wrap models in reusable functions/classes

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute!

## Acknowledgments

* [Scikit-learn documentation](https://scikit-learn.org/)
* The Iris dataset by Ronald A. Fisher
* Kaggle for providing an accessible data science platform
