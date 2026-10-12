#  METRICS HELPER

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

import numpy as np

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray | None, classes: list) -> dict:
    """
    Returns accuracy, per-class sensitivity, macro-avg specificity, F1, AUC.
    """
    n_cls = len(classes)
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm    = confusion_matrix(y_true, y_pred, labels=classes)

    # per-class sensitivity & specificity from CM
    sensitivities  = []
    specificities  = []
    for idx in range(n_cls):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn + 1e-10)
        spec = tn / (tn + fp + 1e-10)
        sensitivities.append(sens)
        specificities.append(spec)

    macro_sens = float(np.mean(sensitivities))
    macro_spec = float(np.mean(specificities))

    # AUC
    auc = None
    if y_prob is not None and n_cls == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    elif y_prob is not None and n_cls > 2:
        try:
            auc = roc_auc_score(
                y_true, y_prob,
                multi_class="ovr", average="macro"
            )
        except Exception:
            auc = None

    return {
        "accuracy":    round(acc,        4),
        "sensitivity": round(macro_sens, 4),   # macro recall
        "specificity": round(macro_spec, 4),
        "f1_macro":    round(f1,         4),
        "auc_roc":     round(auc, 4) if auc is not None else None,
        "per_class_sensitivity": {
            str(classes[i]): round(sensitivities[i], 4) for i in range(n_cls)
        },
        "per_class_specificity": {
            str(classes[i]): round(specificities[i], 4) for i in range(n_cls)
        },
        "confusion_matrix": cm.tolist(),
    }
