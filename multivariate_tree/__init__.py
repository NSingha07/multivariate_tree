from .model import (
    Node,
    compute_mean_and_sse,
    find_best_split,
    build_tree,
    predict,
    print_tree,
    mean_squared_error,
    r2_score,
    nrmse,
    evaluate_model,
)

from .visualization import (
    generate_leaf_histogram,
    visualize_tree_graphviz,
    plot_maxdepth_vs_r2,  # Newly added function
)
