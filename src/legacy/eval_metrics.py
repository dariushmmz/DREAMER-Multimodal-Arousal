import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

import torch
import torch.nn.functional as F

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Add these for consistent saving
plt.rcParams['savefig.facecolor'] = plt.rcParams['figure.facecolor']  # Use same as display
plt.rcParams['savefig.edgecolor'] = plt.rcParams['figure.edgecolor']
plt.rcParams['savefig.transparent'] = False  # Set to True if you want transparent background

# METRICS
def calculate_metrics(y_true, y_pred):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        )
    }

    return metrics


# PLOTS
def plot_confusion_matrix(cm,
                          class_names,
                          save_path,
                          normalize=False):

    if normalize:
        cm = cm.astype(np.float32) / (
            cm.sum(axis=1, keepdims=True) + 1e-8
        )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    title = "Normalized Confusion Matrix" \
        if normalize else "Confusion Matrix"

    plt.title(title)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_metrics_bar(metrics, save_path):

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(8, 5))

    bars = plt.bar(metric_names, metric_values)

    for bar, value in zip(bars, metric_values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.4f}",
            ha='center',
            fontsize=10
        )

    plt.ylim(0, 1.05)

    plt.ylabel("Score")
    plt.title("Evaluation Metrics")

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_roc_curve(y_true,
                   y_probs,
                   n_classes,
                   class_names,
                   save_path):

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):

        binary_true = (np.array(y_true) == i).astype(int)

        fpr, tpr, _ = roc_curve(binary_true, y_probs[:, i])

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_names[i]} (AUC = {roc_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_precision_recall_curve(y_true,
                                y_probs,
                                n_classes,
                                class_names,
                                save_path):

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):

        binary_true = (np.array(y_true) == i).astype(int)

        precision, recall, _ = precision_recall_curve(
            binary_true,
            y_probs[:, i]
        )

        pr_auc = auc(recall, precision)

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{class_names[i]} (AUC = {pr_auc:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


# MAIN EVALUATION FUNCTION
def evaluate_model(model,
                   test_loader,
                   save_root,
                   class_names=None,
                   device='cuda'):

    os.makedirs(save_root, exist_ok=True)

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for eeg_batch, ecg_batch, hrv_batch, labels in test_loader:

            eeg_batch = eeg_batch.to(device)
            ecg_batch = ecg_batch.to(device)
            hrv_batch = hrv_batch.to(device)

            labels = labels.to(device)

            outputs = model(
                eeg_batch,
                ecg_batch,
                hrv_batch
            )

            probs = F.softmax(outputs, dim=1)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # METRICS
    metrics = calculate_metrics(
        all_labels,
        all_preds
    )

    print("\n========== TEST RESULTS ==========")

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # CLASSIFICATION REPORT
    if class_names is None:

        num_classes = len(np.unique(all_labels))

        class_names = [
            f"Class_{i}" for i in range(num_classes)
        ]

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )

    print("\nClassification Report:\n")
    print(report)

    report_path = os.path.join(
        save_root,
        "classification_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)


    print()
    print()

    # CONFUSION MATRIX
    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=os.path.join(
            save_root,
            "confusion_matrix.png"
        ),
        normalize=False
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=os.path.join(
            save_root,
            "confusion_matrix_normalized.png"
        ),
        normalize=True
    )

    # METRICS BAR PLOT
    plot_metrics_bar(
        metrics,
        save_path=os.path.join(
            save_root,
            "metrics_barplot.png"
        )
    )

    # ROC CURVE
    plot_roc_curve(
        y_true=all_labels,
        y_probs=all_probs,
        n_classes=len(class_names),
        class_names=class_names,
        save_path=os.path.join(
            save_root,
            "roc_curve.png"
        )
    )

    # PRECISION-RECALL CURVE
    plot_precision_recall_curve(
        y_true=all_labels,
        y_probs=all_probs,
        n_classes=len(class_names),
        class_names=class_names,
        save_path=os.path.join(
            save_root,
            "precision_recall_curve.png"
        )
    )

    # SAVE METRICS JSON
    metrics_path = os.path.join(
        save_root,
        "metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nAll evaluation files saved to:\n{save_root}")

    return metrics, cm