"""
Visualization Module for UKM Live Notulen ASR
Provides comparison charts and graphs for model evaluation.

Visualizations included:
- WER comparison bar charts
- Speed comparison (RTF/latency)
- Accuracy vs speed tradeoff plots
- Code-switching analysis charts
- Comparison tables
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import os


# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


# Color palette for models
MODEL_COLORS = {
    'whisper-base': '#1f77b4',
    'whisper-small': '#ff7f0e',
    'whisper-tiny': '#2ca02c',
    'wav2vec2-large': '#d62728',
    'wav2vec2': '#d62728',
    'mms-1b': '#9467bd',
    'default': '#7f7f7f'
}


def get_model_color(model_name: str) -> str:
    """Get color for a model name."""
    model_lower = model_name.lower()
    for key, color in MODEL_COLORS.items():
        if key in model_lower:
            return color
    return MODEL_COLORS['default']


def plot_wer_comparison(
    results_dict: Dict[str, Dict[str, float]],
    test_sets: Optional[List[str]] = None,
    title: str = "Word Error Rate Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot WER comparison across models and test sets.
    
    Args:
        results_dict: Dict mapping model_name -> {test_set_name -> wer_value}
        test_sets: List of test set names to include (None for all)
        title: Plot title
        save_path: Path to save figure (None to not save)
    
    Returns:
        matplotlib Figure object
    """
    if test_sets is None:
        # Get all test sets from first model
        first_model = list(results_dict.keys())[0]
        test_sets = list(results_dict[first_model].keys())
    
    models = list(results_dict.keys())
    x = np.arange(len(test_sets))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, model in enumerate(models):
        wers = [results_dict[model].get(ts, 0) * 100 for ts in test_sets]  # Convert to percentage
        bars = ax.bar(x + i * width, wers, width, label=model, color=get_model_color(model))
        
        # Add value labels on bars
        for bar, wer in zip(bars, wers):
            ax.annotate(f'{wer:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Test Set')
    ax.set_ylabel('WER (%)')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(test_sets, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, max([max(results_dict[m].values()) for m in models]) * 100 * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_speed_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = ['rtf', 'avg_latency'],
    title: str = "Speed Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot RTF and latency comparison across models.
    
    Args:
        results_dict: Dict mapping model_name -> {metric_name -> value}
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure object
    """
    models = list(results_dict.keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = [results_dict[m].get(metric, 0) for m in models]
        colors = [get_model_color(m) for m in models]
        
        bars = ax.bar(models, values, color=colors)
        
        # Add value labels
        for bar, val in zip(bars, values):
            label = f'{val:.3f}s' if 'latency' in metric else f'{val:.2f}x'
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('RTF' if metric == 'rtf' else 'Latency (seconds)')
        ax.set_title(f"{'Real-Time Factor' if metric == 'rtf' else 'Average Latency'}")
        ax.set_xticklabels(models, rotation=30, ha='right')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_accuracy_speed_tradeoff(
    results_dict: Dict[str, Dict[str, float]],
    wer_key: str = 'wer',
    rtf_key: str = 'rtf',
    title: str = "Accuracy vs Speed Tradeoff",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot WER vs RTF to show accuracy-speed tradeoff.
    
    Lower-left corner is best (low WER, fast inference).
    
    Args:
        results_dict: Dict mapping model_name -> {metric_name -> value}
        wer_key: Key for WER metric
        rtf_key: Key for RTF metric
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, metrics in results_dict.items():
        wer = metrics.get(wer_key, 0) * 100  # Percentage
        rtf = metrics.get(rtf_key, 0)
        
        ax.scatter(rtf, wer, s=200, c=[get_model_color(model_name)], 
                  label=model_name, alpha=0.8, edgecolors='black', linewidth=1.5)
        
        # Add model name label
        ax.annotate(model_name,
                   xy=(rtf, wer),
                   xytext=(5, 5),
                   textcoords="offset points",
                   fontsize=10,
                   alpha=0.9)
    
    ax.set_xlabel('Real-Time Factor (lower is faster)', fontsize=12)
    ax.set_ylabel('WER % (lower is better)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add "ideal region" indicator
    ax.axhline(y=min([m.get(wer_key, 0) for m in results_dict.values()]) * 100 * 1.5, 
               color='green', linestyle='--', alpha=0.3, label='Target WER')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.3, label='Real-time threshold')
    
    # Add quadrant labels
    ax.text(0.02, 0.98, 'FAST & ACCURATE\n(Ideal)', transform=ax.transAxes,
           fontsize=10, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_code_switching_analysis(
    results_dict: Dict[str, Dict[str, float]],
    title: str = "Code-Switching Accuracy",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot code-switching accuracy comparison.
    
    Args:
        results_dict: Dict mapping model_name -> {'cs_accuracy' -> value}
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure object
    """
    models = list(results_dict.keys())
    cs_accs = [results_dict[m].get('cs_accuracy', 0) * 100 for m in models]
    colors = [get_model_color(m) for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, cs_accs, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, acc in zip(bars, cs_accs):
        ax.annotate(f'{acc:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Code-Switching Accuracy (%)')
    ax.set_title(title)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylim(0, 100)
    
    # Add reference line
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Target (80%)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_error_breakdown(
    results_dict: Dict[str, Dict[str, int]],
    title: str = "Error Type Breakdown",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot stacked bar chart of error types (substitutions, deletions, insertions).
    
    Args:
        results_dict: Dict mapping model_name -> error counts
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure object
    """
    models = list(results_dict.keys())
    
    substitutions = [results_dict[m].get('substitutions', 0) for m in models]
    deletions = [results_dict[m].get('deletions', 0) for m in models]
    insertions = [results_dict[m].get('insertions', 0) for m in models]
    
    x = np.arange(len(models))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.bar(x, substitutions, label='Substitutions', color='#ff6b6b')
    ax.bar(x, deletions, bottom=substitutions, label='Deletions', color='#4ecdc4')
    ax.bar(x, insertions, bottom=np.array(substitutions) + np.array(deletions), 
           label='Insertions', color='#45b7d1')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Error Count')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_comparison_table(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = ['wer', 'cer', 'rtf', 'avg_latency', 'cs_accuracy'],
    save_path: Optional[str] = None
) -> str:
    """
    Generate formatted markdown comparison table.
    
    Args:
        results_dict: Dict mapping model_name -> {metric_name -> value}
        metrics: List of metrics to include
        save_path: Path to save table
    
    Returns:
        Markdown-formatted table string
    """
    models = list(results_dict.keys())
    
    # Build header
    header = "| Model | " + " | ".join(m.upper().replace('_', ' ') for m in metrics) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"
    
    # Build rows
    rows = []
    for model in models:
        values = []
        for metric in metrics:
            val = results_dict[model].get(metric, 0)
            if 'accuracy' in metric or metric in ['wer', 'cer']:
                values.append(f"{val * 100:.1f}%")
            elif 'latency' in metric:
                values.append(f"{val:.3f}s")
            elif metric == 'rtf':
                values.append(f"{val:.2f}x")
            else:
                values.append(f"{val:.3f}")
        
        row = f"| {model} | " + " | ".join(values) + " |"
        rows.append(row)
    
    table = header + "\n" + separator + "\n" + "\n".join(rows)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("# Model Comparison Results\n\n")
            f.write(table)
    
    return table


def plot_training_curves(
    training_history: Dict[str, List[float]],
    title: str = "Training Progress",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training loss and evaluation metrics over time.
    
    Args:
        training_history: Dict with 'loss', 'eval_wer', etc. lists
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    if 'loss' in training_history:
        axes[0].plot(training_history['loss'], label='Training Loss', color='blue')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Evaluation WER
    if 'eval_wer' in training_history:
        axes[1].plot(training_history['eval_wer'], label='Eval WER', color='red', marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('WER')
        axes[1].set_title('Evaluation WER')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# Demo
def demo_visualizations():
    """Generate demo visualizations with synthetic data."""
    print("Generating demo visualizations...")
    
    # Synthetic results
    results = {
        'whisper-base': {
            'wer': 0.15, 'cer': 0.08, 'rtf': 0.35, 'avg_latency': 0.42,
            'cs_accuracy': 0.72, 'substitutions': 45, 'deletions': 20, 'insertions': 15
        },
        'whisper-tiny': {
            'wer': 0.22, 'cer': 0.12, 'rtf': 0.12, 'avg_latency': 0.15,
            'cs_accuracy': 0.65, 'substitutions': 60, 'deletions': 35, 'insertions': 25
        },
        'wav2vec2-large': {
            'wer': 0.18, 'cer': 0.10, 'rtf': 0.25, 'avg_latency': 0.30,
            'cs_accuracy': 0.68, 'substitutions': 50, 'deletions': 25, 'insertions': 20
        }
    }
    
    # Generate plots
    os.makedirs('outputs/comparison_results', exist_ok=True)
    
    plot_wer_comparison(
        {'whisper-base': {'fleurs_id': 0.12, 'synthetic_cs': 0.18, 'ukm_real': 0.22},
         'whisper-tiny': {'fleurs_id': 0.18, 'synthetic_cs': 0.25, 'ukm_real': 0.30},
         'wav2vec2-large': {'fleurs_id': 0.15, 'synthetic_cs': 0.20, 'ukm_real': 0.25}},
        save_path='outputs/comparison_results/wer_comparison.png'
    )
    
    plot_accuracy_speed_tradeoff(results, save_path='outputs/comparison_results/accuracy_speed.png')
    
    plot_code_switching_analysis(results, save_path='outputs/comparison_results/cs_accuracy.png')
    
    # Print table
    print("\nComparison Table:")
    print(generate_comparison_table(results))
    
    print("\nDemo visualizations saved to outputs/comparison_results/")


if __name__ == "__main__":
    demo_visualizations()
