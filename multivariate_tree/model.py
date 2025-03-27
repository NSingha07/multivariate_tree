class Node:
    def __init__(self, indices, prediction, sse, split_feature=None, split_value=None, split_categories=None):
        self.indices = indices
        self.prediction = prediction
        self.sse = sse
        self.split_feature = split_feature
        self.split_value = split_value
        self.split_categories = split_categories
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None


def compute_mean_and_sse(indices, Y):
    """Compute mean vector and within-group SSE for the given data indices."""
    #print(f"\nComputing mean and SSE for indices: {indices}")

    n = len(indices)
    d = len(Y[indices[0]])
    mean_vec = [0.0] * d

    # Compute mean vector
    for i in indices:
        for j in range(d):
            mean_vec[j] += Y[i][j]
    mean_vec = [m / n for m in mean_vec]
    #print(f"Mean vector: {mean_vec}")

    # Compute SSE
    sse = 0.0
    for i in indices:
        for j in range(d):
            diff = Y[i][j] - mean_vec[j]
            sse += diff * diff
    #print(f"SSE: {sse}")

    return mean_vec, sse


def find_best_split(indices, X, Y):
    """Finds the best split by minimizing the within-group SSE."""
    #print("\nFinding the best split...")

    _, current_sse = compute_mean_and_sse(indices, Y)
    best_feature, best_value = None, None
    best_left_idx, best_right_idx = None, None
    best_left_sse, best_right_sse = float('inf'), float('inf')

    # Handle missing column names
    if isinstance(X[0], dict):
        feature_names = list(X[0].keys())  # Use dictionary keys as column names
    else:
        feature_names = [f"X{i}" for i in range(len(X[0]))]  # Assign X0, X1, X2, ...

    for feature_index, feature in enumerate(feature_names):
        # Access feature values correctly for both list-based and dict-based X
        unique_values = sorted(set(X[i][feature_index] if isinstance(X[0], list) else X[i][feature] for i in indices))

        for val in unique_values:
            left_idx = [
                i for i in indices if (X[i][feature_index] if isinstance(X[0], list) else X[i][feature]) <= val
            ]
            right_idx = [
                i for i in indices if (X[i][feature_index] if isinstance(X[0], list) else X[i][feature]) > val
            ]

            if len(left_idx) < 2 or len(right_idx) < 2:
                continue

            _, left_sse = compute_mean_and_sse(left_idx, Y)
            _, right_sse = compute_mean_and_sse(right_idx, Y)
            total_sse = left_sse + right_sse

            #print(f"Feature: {feature}, Value: {val}, Left SSE: {left_sse}, Right SSE: {right_sse}, Total SSE: {total_sse}")

            if total_sse < best_left_sse + best_right_sse:
                best_feature, best_value = feature, val
                best_left_idx, best_right_idx = left_idx, right_idx
                best_left_sse, best_right_sse = left_sse, right_sse

    if best_feature is None:
        #print("No valid split found.")
        return None, None, None, (None, None), (None, None)

    #print(f"Best split -> Feature: {best_feature}, Value: {best_value}, SSE Reduction: {current_sse - (best_left_sse + best_right_sse)}")
    return best_feature, best_value, None, (best_left_idx, best_right_idx), (best_left_sse, best_right_sse)


def build_tree(X, Y, indices=None, depth=0, max_depth=None, min_samples_split=2, min_sse_reduction=0.01):
    """Builds the Multivariate Regression Tree with print statements for debugging."""
    if indices is None:
        indices = list(range(len(X)))

    #print(f"\nBuilding tree at depth {depth} with {len(indices)} samples.")

    mean_vec, current_sse = compute_mean_and_sse(indices, Y)

    if (max_depth is not None and depth >= max_depth) or len(indices) < min_samples_split:
        #print(f"Stopping at depth {depth} (Leaf node created).")
        return Node(indices, prediction=mean_vec, sse=current_sse)

    feature, value, _, groups, sse_values = find_best_split(indices, X, Y)
    left_idx, right_idx = groups
    left_sse, right_sse = sse_values

    if feature is None or (current_sse - (left_sse + right_sse)) < min_sse_reduction:
        #print(f"Stopping at depth {depth} due to minimal SSE improvement.")
        return Node(indices, prediction=mean_vec, sse=current_sse)

    #print(f"\nSplitting on {feature} <= {value} at depth {depth}")

    node = Node(indices, prediction=mean_vec, sse=current_sse, split_feature=feature, split_value=value)

    node.left = build_tree(X, Y, left_idx, depth + 1, max_depth, min_samples_split, min_sse_reduction)
    node.right = build_tree(X, Y, right_idx, depth + 1, max_depth, min_samples_split, min_sse_reduction)

    return node


def predict(node, x_sample):
    """Predicts the output for a given sample."""
    if node.is_leaf():
        #print(f"\nReached leaf node. Prediction: {node.prediction}")
        return node.prediction

    feat = node.split_feature
    #print(f"Checking feature {feat}, value: {x_sample[feat]} against split value {node.split_value}")

    if x_sample[feat] <= node.split_value:
        return predict(node.left, x_sample)
    else:
        return predict(node.right, x_sample)


def print_tree(node, indent=""):
    """Prints the tree structure with indentation for readability."""
    if node.is_leaf():
        print(f"{indent}Leaf: n={len(node.indices)}, prediction={node.prediction}")
    else:
        print(f"{indent}Node: if ({node.split_feature} <= {node.split_value})")
        print_tree(node.left, indent + "  ")
        print_tree(node.right, indent + "  ")


from math import sqrt

def mean_squared_error(y_true, y_pred):
    """Computes Mean Squared Error (MSE) for multivariate Y."""
    n_samples = len(y_true)
    n_targets = len(y_true[0])  # Number of target variables
    mse_per_target = [sum((y_true[i][j] - y_pred[i][j]) ** 2 for i in range(n_samples)) / n_samples
                      for j in range(n_targets)]
    return sum(mse_per_target) / n_targets  # Average MSE across all targets

def r2_score(y_true, y_pred):
    """Computes R² score for multivariate Y."""
    n_samples = len(y_true)
    n_targets = len(y_true[0])  # Number of target variables

    mean_y_true = [sum(y_true[i][j] for i in range(n_samples)) / n_samples for j in range(n_targets)]
    ss_total = sum((y_true[i][j] - mean_y_true[j]) ** 2 for i in range(n_samples) for j in range(n_targets))
    ss_residual = sum((y_true[i][j] - y_pred[i][j]) ** 2 for i in range(n_samples) for j in range(n_targets))

    return 1 - (ss_residual / ss_total)

def nrmse(y_true, y_pred):
    """Computes Normalized Root Mean Squared Error (NRMSE) for multivariate Y."""
    n_samples = len(y_true)
    n_targets = len(y_true[0])

    mse_per_target = [sum((y_true[i][j] - y_pred[i][j]) ** 2 for i in range(n_samples)) / n_samples
                      for j in range(n_targets)]
    rmse_per_target = [sqrt(mse) for mse in mse_per_target]

    min_y, max_y = min(min(row) for row in y_true), max(max(row) for row in y_true)
    return sum(rmse_per_target) / (n_targets * (max_y - min_y))  # Normalize RMSE


def evaluate_model(tree, X_test, Y):
    """Evaluates the trained MRT model using print statements."""
    #print("\nEvaluating Model...")

    predictions = [predict(tree, x) for x in X_test]

    mse = mean_squared_error(Y, predictions)
    r2 = r2_score(Y, predictions)
    error_nrmse = nrmse(Y, predictions)

    #print(f"\nEvaluation Results:")
    #print(f"MSE: {mse:.4f}")
    #print(f"R² Score: {r2:.4f}")
    #print(f"NRMSE: {error_nrmse:.4f}")

    return {
        'MSE': mse,
        'R2 Score': r2,
        'NRMSE': error_nrmse
    }
