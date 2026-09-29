## Iris flower classification
 machine learning project that classifies iris flowers into three species
 setosa,versicolor, and virginica  using flower measurement data and scikit-learn classification models.
 ## Project Overview
 This project uses the classic Iris dataset to predict flower species from four numerical features:
Sepal lenght
Sepal width
Petal lenght
Petal width
The dataset contains 150 observations and 5 columns.
All four measurement columns contain 150 non-null numerical values, and the target column species, contains the flower classifications.
## Data set
iris_three_flower_classes.csv
## Example record
sepal_length  sepal_width  petal_length  petal_width  species
5.1           3.5          1.4           0.2           setosa
The dataset includes three target classes:
setosa
versicolor
virginica
## Technologies used
Python
Pandas
Matplotlib
seaborn 
google colab
## Machine learning  workflow
1.Load the dataset using Pandas.
2.Inspect the first and last records.
3.Perform basic exploratory data analysis.
4.Check dataset structure and descriptive statistics.
5.Separate features and target labels.
6.Split the data into training and testing sets.
7.Train classification models.
8.Evaluate model accuracy.
9.Generate confusion matrices.
10.Compare model performance.
## Models
Logistic Regression 

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)

Decision tree

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

## Results
| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 100%     |
| Decision Tree       | 100%     |

The confusion matrix show no errors
| Actual Class | Setosa | Versicolor | Virginica |
| ------------ | ------ | ---------- | --------- |
| Setosa       | 10     | 0          | 0         |
| Versicolor   | 0      | 9          | 0         |
| Virginica    | 0      | 0          | 11        |

## How to run
1.Open the notebook in Google Colab or Jupyter Notebook.
2.Upload iris_three_flower_classes.csv.
3.Run the import and dataset-loading cells.
4.Run the exploratory data analysis cells.
5.Train the Logistic Regression and Decision Tree models.
6.Review the accuracy scores and confusion matrices.
## Conclusion
This project demonstrates a complete beginner-friendly machine learning workflow, from data loading and exploratory analysis to model training and
evaluation. Logistic Regression and Decision Tree models both achieved 100%accuracy on the test set, with confusion matrices confirming 
that no test samples were misclassified.

