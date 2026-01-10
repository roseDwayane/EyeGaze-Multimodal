# 6_Utils: Utility functions and visualization tools

from .visualizers import (
    # Style
    setup_academic_style,
    CONDITION_COLORS,
    CONDITION_PALETTE,
    # Entropy plots
    plot_entropy_boxplot,
    plot_entropy_kde,
    plot_entropy_topomap,
    plot_entropy_correlation,
    plot_entropy_violin,
    plot_entropy_heatmap,
    # Classification plots
    plot_confusion_matrix,
    plot_roc_curves,
    plot_tsne,
    plot_learning_curves,
    plot_metrics_comparison,
    plot_pair_accuracy,
    plot_mechanism_analysis,
)

from .attention_utils import (
    AttentionAnalyzer,
    denormalize_image,
    tensor_to_numpy_image,
)

from .error_analysis import (
    ErrorAnalyzer,
    MechanismAnalyzer,
)

from .learning_curves import (
    LearningCurveAnalyzer,
    compare_training_histories,
)

from .model_comparison import (
    ModelResults,
    MultiModelComparator,
)

__all__ = [
    # Visualizers - Style
    'setup_academic_style',
    'CONDITION_COLORS',
    'CONDITION_PALETTE',
    # Visualizers - Entropy
    'plot_entropy_boxplot',
    'plot_entropy_kde',
    'plot_entropy_topomap',
    'plot_entropy_correlation',
    'plot_entropy_violin',
    'plot_entropy_heatmap',
    # Visualizers - Classification
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_tsne',
    'plot_learning_curves',
    'plot_metrics_comparison',
    'plot_pair_accuracy',
    'plot_mechanism_analysis',
    # Attention Utils
    'AttentionAnalyzer',
    'denormalize_image',
    'tensor_to_numpy_image',
    # Error Analysis
    'ErrorAnalyzer',
    'MechanismAnalyzer',
    # Learning Curves
    'LearningCurveAnalyzer',
    'compare_training_histories',
    # Model Comparison
    'ModelResults',
    'MultiModelComparator',
]
