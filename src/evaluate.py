import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test, model_name, feature_name):
    """
    Evaluate a single model and return metrics.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Specificity
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        spec = 0
        
    # AUC
    auc = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        auc = roc_auc_score(y_test, y_prob)
        
    # Generate confusion matrix plot
    plot_confusion_matrix(cm, model_name, feature_name)
    
    # Classification report
    cr = classification_report(y_test, y_pred, target_names=["Normal", "Glaucoma"], zero_division=0)
    save_classification_report(cr, model_name, feature_name)
    
    return {
        "Model": model_name,
        "Feature": feature_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Specificity": spec,
        "ROC-AUC": auc
    }

def plot_confusion_matrix(cm, model_name, feature_name):
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Glaucoma"], yticklabels=["Normal", "Glaucoma"])
    plt.title(f"Confusion Matrix: {model_name} ({feature_name})")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"outputs/figures/confusion_matrix_{model_name}_{feature_name}.png")
    plt.close()

def save_classification_report(cr, model_name, feature_name):
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    with open("outputs/reports/classification_report.txt", "a") as f:
        f.write(f"=== {model_name} on {feature_name} ===\n")
        f.write(cr)
        f.write("\n\n")

def save_all_results(results_list):
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv("outputs/reports/results.csv", index=False)
    return df
