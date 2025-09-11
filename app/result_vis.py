import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy  as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from io import BytesIO
import base64
import neurokit2 as nk
import math
from typing import List, Dict, Any, Optional, Tuple

#############################################################################
                    # For the plotly graph in  main #
def generate_entropy_plot_data(output_data: List[List[float]]) -> List[Dict[str, Any]]:
    # Calculate entropy for each prediction
    entropies = []
    for prediction_probs in output_data:
        # Filter out zeros to avoid log(0)
        filtered_probs = [p for p in prediction_probs if p > 0]
        if filtered_probs:
            entropy = -sum(p * math.log2(p) for p in filtered_probs)
            entropies.append(entropy)
    
    # Create histogram data for Plotly
    return [{
        'x': entropies,
        'type': 'histogram',
        'marker': {
            'color': 'rgba(70, 130, 180, 0.7)',
            'line': {
                'color': 'rgba(70, 130, 180, 1)',
                'width': 1
            }
        },
        'opacity': 0.7,
        'name': 'Entropy Distribution'
    }]

def generate_violin_plot_data(output_data: List[List[float]], num_classes: int, class_names: List[str]) -> List[Dict[str, Any]]:
    # Transpose output_data to get probabilities by class
    class_probs = [[] for _ in range(num_classes)]
    
    for probs in output_data:
        for i, p in enumerate(probs):
            if i < num_classes:  # Ensure we don't exceed class count
                class_probs[i].append(p)
    
    # Create violin plot data for Plotly
    violin_data = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, probs in enumerate(class_probs):
        if i < len(class_names):  # Ensure we don't exceed class names
            violin_data.append({
                'y': probs,
                'type': 'violin',
                'name': class_names[i],
                'box': {
                    'visible': True
                },
                'meanline': {
                    'visible': True
                },
                'marker': {
                    'color': colors[i % len(colors)]
                }
            })
    
    return violin_data


def create_plot(freq: np.ndarray, h: np.ndarray) -> str:
    """Generate FIR filter response plot and return base64 string."""
    plt.figure(figsize=(6, 4))
    try:
        plt.plot(freq, np.abs(h), 'k--', label='Filter Response')
        plt.axvline(0.3, color='r', linestyle='--', label='Lower Cutoff')
        plt.axvline(15.0, color='r', linestyle='--', label='Upper Cutoff')
        plt.fill_between(freq, np.abs(h), where=(freq >= 0.3) & (freq <= 15.0), 
                        alpha=0.5, label='Passband')
        plt.fill_between(freq, np.abs(h), where=(freq < 0.3) | (freq > 15.0), 
                        alpha=0.5, label='Stopband')
        plt.title('FIR Filter Frequency Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return plot_base64
    finally:
        plt.close()

def create_blink_stats(signal: np.ndarray, sampling_frequency: int) -> str:
    try:
        plt.figure(figsize=(6, 4))
        s, info = nk.eog_process(signal, sampling_rate=sampling_frequency)
        nk.eog_plot(s, info)
        plt.suptitle('Blink Detection', fontsize =15, fontweight = 'bold')
        plt.grid(True)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return plot_base64
    finally:
        plt.close()

def create_table_html(data: List) -> str:
    """Generate HTML table from data."""
    table_html = """
    <table style="border-collapse: collapse; margin: 5px; float: right; width: 750px;">
        <thead>
            <tr>
                <th style="border: 1px solid #ddd; padding: 2px; text-align: center; 
                    font-size: 30px !important; background-color: #f2f2f2;">Attribute</th>
                <th style="border: 1px solid #ddd; padding: 2px; text-align: center; 
                    font-size: 30px !important; background-color: #f2f2f2;">Value</th>
            </tr>
        </thead>
        <tbody>
    """
    for attribute, value in data:
        table_html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 2px; text-align: left; 
                    font-weight: bold; font-size: 25px !important; background-color: #ffffff;">{attribute}</td>
                <td style="border: 1px solid #ddd; padding: 2px; text-align: left; 
                    font-size: 25px !important; background-color: #ffffff;">{value}</td>
            </tr>
        """
    table_html += """
        </tbody>
    </table>
    """
    return table_html


def plot_classification_results(true_label: List[Any], pred_label: List[Any], num_class: int, title='Classification Results') -> plt.Figure:
    """
    Create a figure with two subplots:
    1. Classification metrics heatmap (precision, recall, f1-score)
    2. Confusion matrix
    """
    # Create figure and axes for subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    
    # Plot classification report on first subplot
    plot_classification_report(true_label, pred_label, ax=ax1)
    
    # Plot confusion matrix on second subplot
    plot_confusion_matrix(true_label, pred_label, num_class, ax=ax2)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(true_label: List[Any], pred_label: List[Any], num_class: int, ax: Optional[plt.Axes] = None):
    """
    Plot confusion matrix with proper labels and coloring
    """
    if num_class == 2:
        class_names = ['Wake', 'Sleep']
    elif num_class == 3:
        class_names = ['Wake', 'NREM', 'REM']
    elif num_class == 4:
        class_names = ['Wake', 'Light-Sleep', 'Deep-Sleep', 'REM']
    else:
        class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    if ax is None:
        ax = plt.gca()
        
    # Compute confusion matrix
    cm = confusion_matrix(true_label, pred_label)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                annot_kws={'size': 18})
    
    ax.set_title('Confusion Matrix', fontsize=22, pad=20)
    ax.set_ylabel('True Label',fontsize=20)
    ax.set_xlabel('Predicted Label',fontsize=20)

def plot_classification_report(true_label: List[Any], pred_label: List[Any], ax: Optional[plt.Axes] = None):
    """
    Plot classification report as a heatmap
    """
    if ax is None:
        ax = plt.gca()
    
    # Get classification report
    report = classification_report(true_label, pred_label)
    
    # Parse classification report
    lines = report.split('\n')
    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2: (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2:
            continue
        if 'avg' in t[1].lower() or "accuracy" in t[0].lower():
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    # Convert to numpy array
    plotMat = np.array(plotMat)
    
    # Create heatmap
    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
    
    # Plot heatmap
    im = ax.pcolor(plotMat, cmap='RdBu', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(len(yticklabels)):
        for j in range(len(xticklabels)):
            text = ax.text(j + 0.5, i + 0.5, f'{plotMat[i, j]:.2f}',
                         ha="center", va="center", fontsize= 20,
                         color="black" if plotMat[i, j] > 0.5 else "white")
    
    # Customize the plot
    ax.set_xticks(np.arange(len(xticklabels)) + 0.5)
    ax.set_yticks(np.arange(len(yticklabels)) + 0.5)
    ax.set_xticklabels(xticklabels, fontsize=20)
    ax.set_yticklabels(yticklabels, fontsize=20)
    ax.set_title('Classification Metrics', fontsize=22, pad=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Remove ticks
    ax.tick_params(axis='both', which='both', length=0)
#############################################################################
                # For the visualization in test prediction #
def calculate_tpr_fpr(y_real: List[int], y_pred: List[int]) -> Tuple[float, float]:
    cm = confusion_matrix(y_real, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr

def get_roc_coordinate(y_real: List[int], y_proba: List[float]) -> Tuple[List[float], List[float]]:
    fpr, tpr, _ = roc_curve(y_real, y_proba)
    return fpr.tolist(), tpr.tolist()

def plot_roc_curve(tpr: List[float], fpr: List[float], scatter: bool = True, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        plt.figure(figsize=(6, 4))
        ax = plt.axes()
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x= fpr, y= tpr, ax = ax)
    sns.lineplot(x=[0, 1], y=[0, 1], color="green", ax=ax)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    return ax

def plot_confidence_hist(prediction_output: torch.Tensor):
    # Normalize to sum of 1
    predictions = np.array(prediction_output / prediction_output.sum(dim = 1, keepdim= True))
    max_probs = np.max(predictions, axis= 1)
    plt.figure(figsize=(10, 6))
    plt.his(max_probs, bins=20, alpha=0.7, color='#3498db', edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='0.5 threshold')
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Max Probability (Confidence)')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_class_distribution(predicted_labels: np.ndarray, num_classes: int):
    match num_classes:
        case 2:
            classes = ('0', '1')
            classes_named = ("wake","Sleep")
        case 3:
            classes = ('0', '1', '2')
            classes_named = ("wake","NREM","REM")
        case 4:
            classes = ('0', '1', '2', '3')
            classes_named = ("wake","Light-Sleep","Deep-Sleep","REM")
        case 5:
            classes = ('0', '1', '2', '3', '4')
            classes_named = ("wake","N1","N2","N3","REM")
        case _:
            print("You have set the num classes from 2 to 5 IDIOT????")

    plt.figure(figsize=(10, 6))
    class_counts = np.bincount(predicted_labels, minlength = num_classes)
    bars = plt.bars(classes_named, class_counts, color=sns.color_palette('viridis', num_classes))
    plt.title('Distribution of Predicted Classes')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_entropy(prediction_output: torch.Tensor):
    from scipy.stats import entropy
    predictions = np.array(prediction_output / prediction_output.sum(dim = 1, keepdim= True))
    prediction_entropy = entropy(predictions, axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(prediction_entropy, bins=20, alpha=0.7, color='#2ecc71', edgecolor='black')
    plt.title('Distribution of Prediction Entropy (Uncertainty)')
    plt.xlabel('Entropy (Higher = More Uncertain)')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_violin(prediction_output: torch.Tensor, predicted_labels: np.ndarray, num_classes: int):
    match num_classes:
        case 2:
            classes = ('0', '1')
            classes_named = ("wake","Sleep")
        case 3:
            classes = ('0', '1', '2')
            classes_named = ("wake","NREM","REM")
        case 4:
            classes = ('0', '1', '2', '3')
            classes_named = ("wake","Light-Sleep","Deep-Sleep","REM")
        case 5:
            classes = ('0', '1', '2', '3', '4')
            classes_named = ("wake","N1","N2","N3","REM")
        case _:
            print("You have set the num classes from 2 to 5 IDIOT????")

    predictions = np.array(prediction_output / prediction_output.sum(dim = 1, keepdim= True))
    plt.figure(figsize=(12, 7))
    data_by_predicted_class = [predictions[predicted_labels == i, i] for i in range(num_classes)]
    violin_parts = plt.violinplot(data_by_predicted_class, showmeans=True)
    plt.xticks(range(1, num_classes + 1), classes_named)
    plt.title('Confidence Distribution by Predicted Class')
    plt.xlabel('Predicted Class')
    plt.ylabel('Confidence (Probability)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_3d_scatter(prediction_output: torch.Tensor, predicted_labels: np.ndarray, num_classes: int):
    match num_classes:
        case 2:
            classes = ('0', '1')
            classes_named = ("wake","Sleep")
        case 3:
            classes = ('0', '1', '2')
            classes_named = ("wake","NREM","REM")
        case 4:
            classes = ('0', '1', '2', '3')
            classes_named = ("wake","Light-Sleep","Deep-Sleep","REM")
        case 5:
            classes = ('0', '1', '2', '3', '4')
            classes_named = ("wake","N1","N2","N3","REM")
        case _:
            print("You have set the num classes from 2 to 5 IDIOT????")
    predictions = np.array(prediction_output / prediction_output.sum(dim = 1, keepdim= True))

    df = pd.DataFrame(predictions, columns= classes_named)
    sample_indice = np.arange(len(df))
    df['predicted_class'] = predicted_labels
    df['confidence'] = np.max(predictions, axis=1)
    df['sample_id'] = sample_indice

    fig = px.scatter_3d(df, x='wake', y='Sleep', z='REM',
                        color='predicted_class', size='confidence',
                        color_continuous_scale=px.colors.qualitative.Bold,
                        hover_data=['sample_id', 'confidence'])
    fig.update_layout(
        title='3D Visualization of Prediction Probabilities',
        scene=dict(
            xaxis_title=f'Prob. Wake',
            yaxis_title=f'Prob. Sleep',
            zaxis_title=f'Prob. REM',
        )
    )
    fig.show()

def probability_roc_curve(test_tensor: torch.Tensor, predicted_labels: np.ndarray,
                          probabilities: np.ndarray, num_classes: int):
    plt.figure(figsize=(12, 8))
    bins = [i / 20 for i in range(20)] + [1]
    roc_auc_ovr = {}
    match num_classes:
        case 2:
            classes = ('0', '1')
            classes_named = ("wake","Sleep")
        case 3:
            classes = ('0', '1', '2')
            classes_named = ("wake","NREM","REM")
        case 4:
            classes = ('0', '1', '2', '3')
            classes_named = ("wake","Light-Sleep","Deep-Sleep","REM")
        case 5:
            classes = ('0', '1', '2', '3', '4')
            classes_named = ("wake","N1","N2","N3","REM")
        case _:
            print("You have set the num classes from 2 to 5 IDIOT????")
    
    for i, c in enumerate(classes):
        df_aux = pd.DataFrame(test_tensor.reshape(test_tensor.shape[0], -1))
        df_aux['class'] = [1 if str(y) == c else 0 for y in predicted_labels]
        df_aux['prob'] = probabilities[:, i]
        df_aux = df_aux.reset_index(drop=True)

        fig, axes = plt.subplots(2, 1, figsize=(6, 8))
        # Create a histogram for each class for probability
        sns.histplot(x="prob", data=df_aux, hue='class', ax=axes[0], bins=bins, multiple="stack")
        axes[0].set_title(f"P(x = {classes_named[i]})")
        axes[0].legend([f"Class: {classes_named[i]}", "Rest"])
        axes[0].set_xlabel(f"P(x = {c})")
        # Create roc curve for each class
        tpr, fpr = get_roc_coordinate(df_aux['class'], df_aux['prob'])
        plot_roc_curve(tpr, fpr, scatter=False, ax=axes[1])
        axes[1].set_title(f"ROC Curve OvR of class {classes_named[i]}")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_ylim(-0.05, 1.05)

        roc_auc_ovr[c] = roc_auc_score(df_aux["class"], df_aux["prob"])
        plt.tight_layout()
        plt.show()
