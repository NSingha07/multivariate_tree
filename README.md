

```markdown
# Multivariate Regression Tree (MRT) Package

This package implements methods for constructing, evaluating, and visualizing **Multivariate Regression Trees (MRT)**. MRT is a statistical technique developed by De'ath (2002) for exploring, describing, and predicting relationships between multispecies data and environmental characteristics.

Multivariate regression trees form clusters of sites by repeatedly splitting the data with simple rules based on environmental variables. Each split minimizes the dissimilarity of sites within clusters, and the final tree provides both a predictive model and an ecological interpretation of species assemblages relative to environmental conditions.

## Background

Multivariate regression trees (MRT), as described in:

> **De'ath, G. (2002).** *Multivariate Regression Trees: A New Technique for Modeling Species-Environment Relationships.* **Ecology, 83(4),** 1105–1117.  
> DOI: [10.2307/3071917](https://doi.org/10.2307/3071917)

are particularly useful for:
- Analyzing complex ecological data (with imbalances, missing values, nonlinear relationships, and high-order interactions).
- Predicting species composition based solely on environmental data.
- Providing a graphical representation (tree) where each branch corresponds to a species assemblage and its associated habitat.

## Installation

Clone the repository and install the package. You can install it in editable mode for development.

```bash
git clone https://github.com/yourusername/multivariate_tree.git
cd multivariate_tree
pip install -e .
```

## Package Structure

The repository is organized as follows:

```
multivariate_tree/
├── setup.py
├── README.md
├── LICENSE
└── multivariate_tree
    ├── __init__.py
    ├── model.py
    └── visualization.py
```

- **model.py** contains the core functions for building and evaluating the MRT model.
- **visualization.py** contains functions for visualizing the MRT (including leaf histograms and tree graphs) as well as plotting performance metrics (e.g., max depth vs R²).

## Usage Examples

Below are several code examples that demonstrate how to use the various functionalities of the package.

### 1. Building and Evaluating an MRT Model

```python
# Import core functions
from multivariate_tree import build_tree, evaluate_model, print_tree

# Example: assume you have your data loaded as NumPy arrays or pandas DataFrames.
# For demonstration, we create sample data:
import numpy as np

# Create synthetic data for Y (response variables) and X (predictors)
# Here Y has 3 target variables and X has some environmental predictors.
Y = np.array([
    [0.1910978, 1.5642915, 1.3534104],
    [0.5743539, 1.4040023, 0.7107540],
    [0.0,       0.0,       1.6401854],
    [0.7523856, 2.2506030, 1.1379542],
    [1.0987933, 1.3578862, 0.2126343],
    [0.7835680, 2.9186479, 1.8296707],
    [1.8139399, 1.5591685, 0.0],
    [0.9436658, 1.5667416, 0.2651598],
    [2.0875316, 1.4574247, 1.1136134],
    [1.7829821, 2.0335457, 1.7756639],
    [3.0812930, 0.6086476, 1.1863433],
    [2.8692957, 3.0200150, 0.3664296]
])

# X can be a list of lists or a NumPy array; here we include a categorical variable in the last column.
X = [
    [1.02, 6.9, 'C'],
    [1.08, 6.7, 'A'],
    [1.14, 5.9, 'C'],
    [1.32, 7.9, 'C'],
    [1.38, 6.6, 'B'],
    [1.56, 8.2, 'C'],
    [1.63, 7.0, 'B'],
    [1.73, 8.1, 'B'],
    [1.84, 6.5, 'A'],
    [1.86, 6.7, 'C'],
    [2.82, 6.9, 'B'],
    [2.89, 8.4, 'A']
]

# Build the MRT model with a maximum depth of 4.
tree = build_tree(X, Y, max_depth=4)

# Evaluate the model on a test dataset (for simplicity, using the same data here)
results = evaluate_model(tree, X, Y)

# Print the tree structure
print_tree(tree)
```

### 2. Visualizing the MRT Tree

```python
# Import visualization functions
from multivariate_tree import visualize_tree_graphviz

# Visualize the tree using Graphviz
dot = visualize_tree_graphviz(tree)
# Render and display the tree as a PNG file (this will save the file as "mrt_tree.png")
dot.render("mrt_tree", format="png", cleanup=True)
```

### 3. Plotting Maximum Depth vs R² Score

This function allows you to evaluate model performance over a range of maximum depths.

```python
# Import the plotting function
from multivariate_tree import plot_maxdepth_vs_r2
from sklearn.model_selection import train_test_split

# Split your data into training and test sets (using scikit-learn for convenience)
X_array = np.array(X)  # Convert X to a NumPy array if needed
X_train, X_test, Y_train, Y_test = train_test_split(X_array, Y, test_size=0.3, random_state=42)

# Define a range of maximum depths to test
max_depth_values = [2, 3, 4, 5, 6]

# Plot max depth vs R² Score
plot_maxdepth_vs_r2(X_train, Y_train, X_test, Y_test, max_depth_values)
```

### 4. Generating Leaf Histograms

Each leaf node in the tree can have an associated histogram of target values. This function generates and saves the histogram images.

```python
# Import the leaf histogram generator
from multivariate_tree import generate_leaf_histogram

# Example: generate a histogram for a leaf node with sample prediction values.
leaf_values = [1.0, 2.5, 0.5]  # Replace with the actual prediction vector from a leaf node
leaf_id = 0
hist_path = generate_leaf_histogram(leaf_values, leaf_id)
print("Leaf histogram saved at:", hist_path)
```

## Full Strength of the Package

- **Model Building:** Use `build_tree` to recursively construct a multivariate regression tree.
- **Prediction:** Use `predict` (available in the package) to predict new samples based on the built tree.
- **Evaluation:** Use `evaluate_model` to compute metrics such as Mean Squared Error (MSE), R² Score, and Normalized Root Mean Squared Error (NRMSE).
- **Visualization:**  
  - Use `visualize_tree_graphviz` for a detailed graphical representation of the tree.
  - Use `generate_leaf_histogram` to create histograms for leaf nodes.
  - Use `plot_maxdepth_vs_r2` to analyze how changing the maximum depth of the tree affects model performance.

For more detailed documentation on each function, please refer to the inline comments in the source code files (`model.py` and `visualization.py`).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **De'ath, G. (2002)** for developing the original MRT method.
- The example dataset provided in Appendix 1 and other resources used for testing.

## Contact

For questions or suggestions, please contact **Nitul Singha** at [nitulsingha07@gmail.com](mailto:nitulsingha07@gmail.com).

---


