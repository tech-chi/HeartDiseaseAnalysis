import pandas as pd
import GWCutilities as util
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

# Create a variable to read the dataset
try:
    df = pd.read_csv("heartDisease_2020_sampling.csv")
except FileNotFoundError:
    print("Error: The file 'heartDisease_2020_sampling.csv' was not found.")
    exit()

print("We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it:\n")
print(df.head())

input("\nPress Enter to continue.\n")

# Data Cleaning
# Label encode the dataset
df = util.labelEncoder(df, ["HeartDisease", "GenHealth"])

print("\nHere is a preview of the dataset after label encoding:\n")
print(df.head())

input("\nPress Enter to continue.\n")

# One hot encode the dataset
df = pd.get_dummies(df, drop_first=True)

print("\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis:\n")
print(df.head())

input("\nPress Enter to continue.\n")

# Create and train Decision Tree Model
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.head())

# Initialize variables for max_depth and accuracy comparisons
best_max_depth = None
best_accuracy_diff = float('inf')  # To track the minimum accuracy difference
best_test_acc = 0

# Loop through different max_depth values to find the optimal one
for depth in range(1, 21):  # Try max_depth values from 1 to 20
    clf = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    # Test the model with the testing data set and print accuracy score
    test_predictions = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_predictions)

    # Test the model with the training data set and prints accuracy score
    train_predictions = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_predictions)

    # Check if accuracy is within 5% of each other
    accuracy_diff = abs(train_acc - test_acc)

    if accuracy_diff <= 0.05 and test_acc > best_test_acc:  # Maximize accuracy while ensuring difference is within 5%
        best_max_depth = depth
        best_accuracy_diff = accuracy_diff
        best_test_acc = test_acc
        best_train_acc = train_acc

print(f"Best max_depth value: {best_max_depth}")
print(f"Training accuracy: {best_train_acc:.2f}")
print(f"Testing accuracy: {best_test_acc:.2f}")

# Print the confusion matrix
cm = confusion_matrix(y_test, test_predictions, labels=[1, 0])
print("The confusion matrix of the tree is : ")
print(cm)

# Print a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")
util.printTree(clf, X.columns)

leaf_nodes = clf.apply(X_test)  
correct_classified = (test_predictions == y_test) & (y_test == 1)  
leaf_node_counts = {node: sum(correct_classified[leaf_nodes == node]) for node in set(leaf_nodes)}


max_leaf_node = max(leaf_node_counts, key=leaf_node_counts.get)
print(f"Leaf node {max_leaf_node} has the highest count of correctly classified heart disease patients.")

patients_in_leaf = X_test[leaf_nodes == max_leaf_node]
print("\nFeatures of patients in this leaf node:")
print(patients_in_leaf.head())
