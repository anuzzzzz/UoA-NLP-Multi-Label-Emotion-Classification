import pandas as pd
from sklearn.metrics import f1_score, classification_report
import argparse

def evaluate_predictions(true_labels_path: str, predictions_path: str, labels: list):
    """
    Compares true labels with predicted labels and prints F1 scores.

    Args:
        true_labels_path (str): Path to the CSV file with true labels (e.g., dev.csv).
        predictions_path (str): Path to the CSV file with model predictions (e.g., submission.csv).
        labels (list): List of emotion label column names.
    """
    try:
        true_df = pd.read_csv(true_labels_path)
        pred_df = pd.read_csv(predictions_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both CSV files exist at the specified paths.")
        return

    if not all(label in true_df.columns for label in labels):
        print(f"Error: Not all labels {labels} found in true_df columns: {true_df.columns}")
        return
    if not all(label in pred_df.columns for label in labels):
        print(f"Error: Not all labels {labels} found in pred_df columns: {pred_df.columns}")
        return
    
    if len(true_df) != len(pred_df):
        print(f"Error: Row count mismatch. True labels: {len(true_df)}, Predictions: {len(pred_df)}")
        return

    # Ensure 'text' column exists and align dataframes by it if necessary,
    # or assume rows are in the same order. For simplicity, we'll assume order.
    # For robustness, you might want to merge on 'text' or an ID if available.
    # true_df = true_df.sort_values(by="text").reset_index(drop=True)
    # pred_df = pred_df.sort_values(by="text").reset_index(drop=True)


    y_true = true_df[labels].values
    y_pred = pred_df[labels].values

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    print(f"\nMicro F1 Score: {micro_f1:.4f}\n")

    print("Per-label F1 Scores:")
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, label in enumerate(labels):
        print(f"{label}_F1: {per_label_f1[i]:.4f}")

    print("\nClassification Report (per label):")
    # Note: classification_report expects 1D arrays for target_names,
    # so we iterate for multi-label or use a more complex multi-label report.
    # For a simple view, we can show individual reports or just the F1s above.
    # The f1_score with average=None already gives us the per-label scores.
    # For a full report:
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against true labels.")
    parser.add_argument("--true_labels_csv", type=str, default="dev.csv",
                        help="Path to the CSV file containing the true labels (e.g., dev.csv).")
    parser.add_argument("--predictions_csv", type=str, default="submission.csv",
                        help="Path to the CSV file containing the model's predictions.")
    
    # You need to define the emotion labels your model predicts
    # These should match the column names in your CSV files
    # These are the labels from your project description
    emotion_labels = ["admiration", "amusement", "gratitude", "love", "pride", "relief", "remorse"]
    
    args = parser.parse_args()

    evaluate_predictions(args.true_labels_csv, args.predictions_csv, emotion_labels)