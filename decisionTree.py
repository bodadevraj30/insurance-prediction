import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Dataset
df = pd.read_csv('train.csv')

# Feature Selection
X = df.drop('is_claim', axis=1)
y = df['is_claim']

# Decision Tree Functions

def decision_tree_predict(x, tree):
    if isinstance(tree, np.int64):  # Leaf node
        return tree
    else:
        feature_index, threshold, left, right = tree
        if x[feature_index] <= threshold:
            return decision_tree_predict(x, left)
        else:
            return decision_tree_predict(x, right)

def decision_tree_train(X, y, depth=0, max_depth=None):
    if depth == max_depth or len(set(y)) == 1:
        return np.bincount(y).argmax()  # Return the majority class
    else:
        feature_index, threshold = find_best_split(X, y)
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices
        left = decision_tree_train(X[left_indices], y[left_indices], depth + 1, max_depth)
        right = decision_tree_train(X[right_indices], y[right_indices], depth + 1, max_depth)
        return [feature_index, threshold, left, right]

def find_best_split(X, y):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None

    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices = X[:, feature_index] <= threshold
            right_indices = ~left_indices
            gini = calculate_gini_impurity(y[left_indices], y[right_indices])
            if gini < best_gini:
                best_gini = gini
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

def calculate_gini_impurity(y1, y2):
    size_y1 = len(y1)
    size_y2 = len(y2)
    total_size = size_y1 + size_y2
    gini = (size_y1 / total_size) * gini_impurity(y1) + (size_y2 / total_size) * gini_impurity(y2)
    return gini

def gini_impurity(y):
    if len(y) == 0:
        return 0
    p = np.sum(y) / len(y)
    return 1 - p**2 - (1 - p)**2


# Initialize parameters
k_values = range(2, 11)  # Test k from 2 to 10
best_k = None
best_accuracy = 0
results = {}

# Open a file for writing output
with open('./output/output_dec.txt', 'w') as output_file:
    # Implement manual k-fold cross-validation and accuracy calculation
    for k in k_values:
        np.random.seed(42)  # For reproducibility
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        fold_size = len(X) // k
        accuracies = []

        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

            # Convert data to numpy arrays for simplicity
            X_train_np = X_train.to_numpy()
            y_train_np = y_train.to_numpy()

            # Train the model
            tree = decision_tree_train(X_train_np, y_train_np, max_depth=3)

            # Make Predictions on the Test Set
            X_test_np = X_test.to_numpy()
            y_pred = np.array([decision_tree_predict(X_test_np[i], tree) for i in range(len(X_test_np))])

            # Calculate accuracy manually
            accuracy = np.mean(y_pred == y_test.to_numpy())
            accuracies.append(accuracy)

        # Store results
        results[k] = accuracies

        # Write outputs to the file
        output_file.write(f"Results for K={k}:\n")
        output_file.write(f"Accuracies: {np.mean(accuracies)}\n\n")

        # Update best k and accuracy
        if np.mean(accuracies) > best_accuracy:
            best_k = k
            best_accuracy = np.mean(accuracies)

# Print results for the best k
print(f"Best K: {best_k}")
print(f"Best Accuracy: {best_accuracy}")


# Extract k values and corresponding accuracies for plotting
k_values = list(results.keys())
accuracies = list(np.mean(results.values()))
accuracies = np.mean(list(results.values()))

# Plot the results
plt.plot(k_values, accuracies, marker='o')
plt.title('Cross-validation results for different values of k')
plt.xlabel('Number of Folds (k)')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.show()