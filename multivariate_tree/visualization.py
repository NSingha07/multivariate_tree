import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
import os
import shutil

# Ensure the histograms directory is clean before saving new histograms
shutil.rmtree("histograms", ignore_errors=True)
os.makedirs("histograms", exist_ok=True)

def generate_leaf_histogram(leaf_values, leaf_id):
    """Generates and saves a histogram for a given leaf node."""
    plt.figure(figsize=(2, 1))  # Small size for embedding

    # Use multiple colors for different targets
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    plt.bar(range(len(leaf_values)), leaf_values, color=colors[:len(leaf_values)])

    # Optional: Add labels to bars
    plt.xticks(range(len(leaf_values)), [f"T{i}" for i in range(len(leaf_values))], fontsize=5)
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
        # Internal split node: Include feature, split value, SSE, and sample count
        node_label = (f"{node.split_feature}\n<= {node.split_value:.2f}\n"
                      f"SSE: {node.sse:.2f}\nSamples: {len(node.indices)}")
        dot.node(str(id(node)), label=node_label, shape="ellipse")

    if parent:
        dot.edge(str(parent), str(id(node)), label=edge_label)

    if node.left:
        visualize_tree_graphviz(node.left, dot, id(node), edge_label="Left", leaf_id=leaf_id)
    if node.right:
        visualize_tree_graphviz(node.right, dot, id(node), edge_label="Right", leaf_id=leaf_id)

    return dot
