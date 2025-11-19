import numpy as np

def evaluate_model(name, y_true, y_pred, num_classes=10, print_confusion=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Confusion matrix: rows = true class, cols = predicted class
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    # True positives, false positives, false negatives per class
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP

    # Avoid division by zero
    precision_per_class = TP / (TP + FP + 1e-8)
    recall_per_class    = TP / (TP + FN + 1e-8)
    f1_per_class        = 2 * precision_per_class * recall_per_class / (
        precision_per_class + recall_per_class + 1e-8
    )

    # Macro averages (average over classes)
    precision_macro = precision_per_class.mean()
    recall_macro    = recall_per_class.mean()
    f1_macro        = f1_per_class.mean()

    # Accuracy (overall)
    accuracy = np.mean(y_true == y_pred)

    if print_confusion:
        print(name)
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion)

    return accuracy, precision_macro, recall_macro, f1_macro, confusion


def print_summary_table(results):
    print("\n\n===== FINAL SUMMARY TABLE =====")
    print(f"{'Model':40s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}")
    print("-" * 80)
    for name, acc, prec, rec, f1 in results:
        print(f"{name:40s} {acc:7.3f} {prec:7.3f} {rec:7.3f} {f1:7.3f}")
