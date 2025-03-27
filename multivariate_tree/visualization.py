import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
import os
import shutil

# Ensure the histograms directory is clean before saving new histograms
shutil.rmtree("histograms", ignore_errors=True)
os.makedirs("histograms", exist_ok=True)

def generate_leaf_histogram(leaf_values, leaf_id):
    """Generates and saves a histogram for a given leaf node, using up to 20 distinct colors.
       If the number of target variables exceeds 20, all bars will be blue.
    """
    plt.figure(figsize=(2, 1))  # Small size for embedding

    # Define a palette of at least 20 distinct colors
    palette = [
        'blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 
        'lime', 'pink', 'brown', 'gray', 'olive', 'teal', 'navy', 'maroon', 
        'silver', 'gold', 'orchid', 'turquoise'
    ]

    num_targets = len(leaf_values)
    
    if num_targets <= 20:
        # Use a unique color for each target
        bar_colors = palette[:num_targets]
    else:
        # If there are more than 20 targets, color them all blue
        bar_colors = ['blue'] * num_targets

    # Create the bar chart
    plt.bar(range(num_targets), leaf_values, color=bar_colors)

    # Optional: Add labels to bars
    plt.xticks(range(num_targets), [f"T{i}" for i in range(num_targets)], fontsize=5)
    plt.yticks([])  # Hide y-axis labels
    plt.title(f"Leaf {leaf_id}", fontsize=6)

    # Save the histogram
    hist_path = f"histograms/leaf_{leaf_id}.png"
    plt.savefig(hist_path, dpi=100, bbox_inches="tight")
    plt.close()
    return hist_path  # Return the file path

def visualize_tree_graphviz(node, dot=None, parent=None, edge_label="", leaf_id=[0]):
    """Recursively creates a Graphviz tree structure with SSE values and histograms at leaf nodes."""
    if dot is None:
        dot = Digraph(format='png')
        dot.attr(size='24,12', dpi='100')  # Increase plot size and resolution

    if node.is_leaf():
        # Leaf node: Generate histogram image
        node_label = f"Leaf {leaf_id[0]}"
        hist_path = generate_leaf_histogram(node.prediction, leaf_id[0])  # Generate and get path
        dot.node(str(id(node)), label=node_label, shape="box", image=hist_path)  # Embed histogram
        leaf_id[0] += 1  # Increment unique leaf counter
    else:
        # Determine the format for split_value: use 2 decimal places if numeric, else just convert to string.
        if isinstance(node.split_value, (int, float)):
            split_val_str = f"{node.split_value:.2f}"
        else:
            split_val_str = str(node.split_value)
            
        # Internal split node: Include feature, split value, SSE, and sample count
        node_label = (f"{node.split_feature}\n<= {split_val_str}\n"
                      f"SSE: {node.sse:.2f}\nSamples: {len(node.indices)}")
        dot.node(str(id(node)), label=node_label, shape="ellipse")

    if parent:
        dot.edge(str(parent), str(id(node)), label=edge_label)

    if node.left:
        visualize_tree_graphviz(node.left, dot, id(node), edge_label="Left", leaf_id=leaf_id)
    if node.right:
        visualize_tree_graphviz(node.right, dot, id(node), edge_label="Right", leaf_id=leaf_id)

    return dot
 
