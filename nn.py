import argparse
import pandas as pd
import numpy as np
import datasets
import transformers
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, create_optimizer
from keras.metrics import F1Score # Used during model compilation/evaluation within Keras
from sklearn.metrics import f1_score # Used for manual threshold tuning calculation
import os # For managing file paths (saving weights, thresholds)
import json # For saving and loading thresholds

# --- Configuration ---
# MODIFICATION: Change model_name
model_name = "roberta-base"  # <--- THE KEY CHANGE
MAX_LENGTH = 64             # Keep at 64 for the first roberta-base run
BATCH_SIZE = 16             # Keep batch size, adjust if OOM errors occur
# MODIFICATION: Consider slightly lower LR for a larger model
LEARNING_RATE = 2e-5        # <--- Try 2e-5 (or stick with 3e-5 if 2e-5 is too slow/doesn't improve)
MAX_EPOCHS = 10             # Max epochs for manual early stopping loop
PATIENCE = 3                # Patience for manual early stopping

# Load tokenizer globally once
tokenizer = AutoTokenizer.from_pretrained(model_name)
# --- End Configuration ---

def tokenize(examples):
    """Tokenizes text data."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="np"
    )

def gather_labels(example, label_names):
    """Formats labels into a list of floats."""
    return {"labels": [tf.cast(example[label], tf.float32) for label in label_names]}

def train(model_path="saved_model_robertabase", train_path="train.csv", dev_path="dev.csv"): # Changed model_path
    """Trains the model with manual early stopping and per-label threshold tuning."""
    print(f"Starting training process...")
    print(f"Model: {model_name}, Max Length: {MAX_LENGTH}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    # Load datasets
    print("Loading datasets...")
    try:
        hf_dataset = datasets.load_dataset("csv", data_files={
            "train": train_path,
            "validation": dev_path
        })
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Define label columns
    if not hf_dataset or "train" not in hf_dataset:
        print("Dataset loading failed or 'train' split not found.")
        return
    try:
        potential_text_cols = {'text', 'comment_text', 'body'}
        label_names = [col for col in hf_dataset["train"].column_names if col.lower() not in potential_text_cols]
        if not label_names:
            raise ValueError("Could not automatically determine label columns.")
        num_labels = len(label_names)
        print(f"Identified {num_labels} label columns: {label_names}")
    except Exception as e:
        print(f"Error identifying label names: {e}")
        return

    print("Processing labels...")
    try:
        hf_dataset = hf_dataset.map(lambda x: gather_labels(x, label_names), batched=False)
    except Exception as e:
        print(f"Error processing labels: {e}")
        return

    print("Tokenizing datasets...")
    try:
        hf_dataset = hf_dataset.map(tokenize, batched=True)
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return

    print("Preparing TensorFlow datasets...")
    try:
        tf_columns = ["input_ids", "attention_mask"]
        train_dataset = hf_dataset["train"].to_tf_dataset(
            columns=tf_columns,
            label_cols="labels",
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        val_dataset = hf_dataset["validation"].to_tf_dataset(
            columns=tf_columns,
            label_cols="labels",
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    except Exception as e:
        print(f"Error creating TensorFlow datasets: {e}")
        return

    print("Setting up optimizer and model...")
    try:
        num_train_steps_per_epoch = len(train_dataset)
        if num_train_steps_per_epoch == 0:
             raise ValueError("Training dataset seems to be empty.")
        num_train_steps = num_train_steps_per_epoch * MAX_EPOCHS
        num_warmup_steps = int(num_train_steps * 0.1) # 10% of total steps for warmup

        optimizer, schedule = create_optimizer(
            init_lr=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            weight_decay_rate=0.01
        )
    except Exception as e:
         print(f"Error setting up optimizer: {e}")
         return

    try:
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            id2label={i: label for i, label in enumerate(label_names)},
            label2id={label: i for i, label in enumerate(label_names)}
        )
    except Exception as e:
         print(f"Error loading pretrained model '{model_name}': {e}")
         return

    f1_metric_name = "f1_micro"
    try:
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[F1Score(average="micro", threshold=0.5, name=f1_metric_name)]
        )
    except Exception as e:
        print(f"Error compiling model: {e}")
        return

    best_val_f1 = -1.0
    patience_counter = 0
    os.makedirs(model_path, exist_ok=True)
    best_weights_path = os.path.join(model_path, "temp_best_weights.weights.h5")

    print("Starting training loop with manual early stopping...")
    print(f"Monitoring 'val_{f1_metric_name}' for improvement. Patience: {PATIENCE} epochs.")

    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
        try:
            history = model.fit(
                train_dataset,
                epochs=1,
                validation_data=val_dataset,
                verbose=1
            )
        except Exception as e:
             print(f"Error during model.fit for epoch {epoch + 1}: {e}")
             break

        try:
            current_val_loss = history.history['val_loss'][0]
            metric_key = f'val_{f1_metric_name}'
            if metric_key not in history.history:
                 print(f"Error: Metric key '{metric_key}' not found in history. Available keys: {history.history.keys()}")
                 eval_results = model.evaluate(val_dataset, verbose=0)
                 metric_index = model.metrics_names.index(f1_metric_name)
                 current_val_f1 = eval_results[metric_index + 1]
                 current_val_loss = eval_results[0]
                 print(f"Used model.evaluate. Loss: {current_val_loss:.4f}, F1: {current_val_f1:.4f}")
            else:
                 current_val_f1 = history.history[metric_key][0]
            print(f"Epoch {epoch + 1}: Validation Loss: {current_val_loss:.4f}, Validation F1 Micro: {current_val_f1:.4f}")
        except Exception as e:
            print(f"Error retrieving metrics for epoch {epoch + 1}: {e}")
            break

        if current_val_f1 > best_val_f1:
            print(f"Validation F1 improved from {best_val_f1:.4f} to {current_val_f1:.4f}. Saving best weights to {best_weights_path}")
            best_val_f1 = current_val_f1
            patience_counter = 0
            try:
                model.save_weights(best_weights_path)
            except Exception as e:
                 print(f"Error saving weights: {e}")
        else:
            patience_counter += 1
            print(f"Validation F1 did not improve ({current_val_f1:.4f} <= {best_val_f1:.4f}). Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break
    print("\nTraining loop finished.")

    if os.path.exists(best_weights_path):
        print(f"Loading best weights from epoch with F1: {best_val_f1:.4f}")
        try:
            model.load_weights(best_weights_path)
        except Exception as e:
            print(f"Error loading best weights from {best_weights_path}: {e}. Using weights from last epoch.")
        try:
            os.remove(best_weights_path)
            print(f"Removed temporary weights file: {best_weights_path}")
        except OSError as e:
            print(f"Warning: Error removing temporary weights file {best_weights_path}: {e}")
    else:
         print("Warning: Best weights file not found. Using weights from the last epoch.")

    print("\nFinding optimal per-label thresholds on validation set...")
    try:
        val_logits = model.predict(val_dataset).logits
        val_probs = tf.sigmoid(val_logits).numpy()
        true_labels = np.concatenate([y.numpy() for x, y in val_dataset], axis=0)

        if true_labels.shape[1] != num_labels:
             raise ValueError(f"Mismatch in true_labels shape ({true_labels.shape[1]}) and num_labels ({num_labels})")

        best_thresholds_arr = np.full(num_labels, 0.5)
        threshold_candidates = np.arange(0.1, 0.9, 0.01)
        current_best_overall_f1 = -1

        initial_preds = (val_probs > 0.5).astype(int)
        current_best_overall_f1 = f1_score(true_labels, initial_preds, average='micro', zero_division=0)
        print(f"Initial Micro F1 with 0.5 thresholds for all labels: {current_best_overall_f1:.4f}")

        for i in range(num_labels):
            best_f1_for_this_label_step = -1
            optimal_threshold_for_this_label_idx = best_thresholds_arr[i]

            for threshold_candidate in threshold_candidates:
                temp_preds = (val_probs > best_thresholds_arr).astype(int)
                temp_preds[:, i] = (val_probs[:, i] > threshold_candidate).astype(int)
                current_f1_micro = f1_score(true_labels, temp_preds, average='micro', zero_division=0)

                if current_f1_micro > best_f1_for_this_label_step:
                    best_f1_for_this_label_step = current_f1_micro
                    optimal_threshold_for_this_label_idx = threshold_candidate
            
            if best_f1_for_this_label_step > current_best_overall_f1:
                print(f"Updating threshold for '{label_names[i]}': {best_thresholds_arr[i]:.2f} -> {optimal_threshold_for_this_label_idx:.2f}. Overall F1 improved to {best_f1_for_this_label_step:.4f}")
                best_thresholds_arr[i] = optimal_threshold_for_this_label_idx
                current_best_overall_f1 = best_f1_for_this_label_step
            else:
                print(f"Threshold for '{label_names[i]}' kept at {best_thresholds_arr[i]:.2f}. Tuning this label did not improve overall F1 beyond {current_best_overall_f1:.4f}")

        final_thresholds_dict_for_log = {name: np.round(thresh, 4) for name, thresh in zip(label_names, best_thresholds_arr)}
        print(f"\nFinal Optimized Per-Label Thresholds: {final_thresholds_dict_for_log}")
        print(f"Final Micro F1 on validation set after per-label tuning: {current_best_overall_f1:.4f}")

        threshold_file_path = os.path.join(model_path, "optimal_thresholds_per_label.json")
        threshold_data_to_save = [{"label": name, "threshold": float(thresh)} for name, thresh in zip(label_names, best_thresholds_arr)]
        with open(threshold_file_path, "w") as f:
            json.dump(threshold_data_to_save, f, indent=2)
        print(f"Per-label thresholds saved to {threshold_file_path}")

    except Exception as e:
        print(f"Error during threshold tuning: {e}")
        print("Skipping threshold tuning. Model will be saved without optimal thresholds.")

    print(f"\nSaving final model and tokenizer to {model_path}...")
    try:
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path) # Save the correct tokenizer
        print("Model and tokenizer saving complete.")
    except Exception as e:
        print(f"Error saving final model/tokenizer: {e}")
    print("Training script finished.")


def predict(model_path="saved_model_robertabase", input_path="dev.csv", output_zip="submission_robertabase.zip"): # Changed defaults
    """Generates predictions using the saved model and optimal thresholds."""
    print(f"Starting prediction process...")
    print(f"Loading model and tokenizer from: {model_path}")

    try:
        # Ensure tokenizer is loaded from the model_path to match the trained model
        local_tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        num_labels = model.config.num_labels
    except Exception as e:
        print(f"Error loading model/tokenizer from {model_path}: {e}")
        return

    threshold_file_path = os.path.join(model_path, "optimal_thresholds_per_label.json")
    optimal_thresholds = np.full(num_labels, 0.5)

    try:
        with open(threshold_file_path, "r") as f:
            loaded_threshold_data = json.load(f)

        if not isinstance(loaded_threshold_data, list) or \
           not all(isinstance(item, dict) and "label" in item and "threshold" in item for item in loaded_threshold_data):
            raise ValueError("Threshold file format is incorrect.")

        if not model.config.label2id:
             raise ValueError("Model config (label2id) is missing. Cannot map thresholds by name.")

        model_label_to_id_map = model.config.label2id
        mapped_count = 0
        for item in loaded_threshold_data:
            label_name = item["label"]
            threshold_value = item["threshold"]
            if label_name in model_label_to_id_map:
                label_idx = model_label_to_id_map[label_name]
                if 0 <= label_idx < num_labels:
                    optimal_thresholds[label_idx] = threshold_value
                    mapped_count +=1
                else:
                    print(f"Warning: Label index {label_idx} for '{label_name}' from model config is out of bounds for num_labels ({num_labels}).")
            else:
                print(f"Warning: Label '{label_name}' from threshold file not found in model.config.label2id. Model knows: {list(model_label_to_id_map.keys())}")
        
        if mapped_count == num_labels:
            print(f"Successfully loaded and mapped all {num_labels} per-label thresholds.")
        else:
            print(f"Warning: Mapped {mapped_count}/{num_labels} thresholds. Unmapped or problematic labels will use 0.5.")
        print(f"Applied thresholds: {optimal_thresholds}")

    except FileNotFoundError:
        print(f"Warning: Threshold file '{threshold_file_path}' not found. Using default 0.5 for all labels.")
    except Exception as e:
        print(f"Error loading or processing thresholds file: {e}. Using default 0.5 for all labels.")

    print(f"Loading prediction data from: {input_path}")
    try:
        df = pd.read_csv(input_path)
        potential_text_cols = {'text', 'comment_text', 'body'}
        text_column_found = None
        for col_name in df.columns:
            if col_name.lower() in potential_text_cols:
                text_column_found = col_name
                break
        if not text_column_found:
            raise ValueError("Input CSV must contain a text column (e.g., 'text', 'comment_text', 'body').")
        if text_column_found != 'text':
            df = df.rename(columns={text_column_found: 'text'})
        df['text'] = df['text'].fillna('').astype(str)
    except Exception as e:
        print(f"Error loading or processing input CSV {input_path}: {e}")
        return

    hf_dataset = datasets.Dataset.from_pandas(df)

    # Use the local_tokenizer loaded from the model_path for prediction
    def tokenize_pred(examples):
        return local_tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH, # Ensure this MAX_LENGTH matches training
            return_tensors="np"
        )

    print("Tokenizing prediction data...")
    try:
        hf_dataset = hf_dataset.map(tokenize_pred, batched=True)
    except Exception as e:
        print(f"Error tokenizing prediction data: {e}")
        return

    tf_columns = ["input_ids", "attention_mask"]
    try:
        cols_to_remove = [col for col in hf_dataset.column_names if col not in tf_columns]
        if cols_to_remove:
            hf_dataset = hf_dataset.remove_columns(cols_to_remove)
        tf_dataset = hf_dataset.to_tf_dataset(
            columns=tf_columns,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    except Exception as e:
        print(f"Error creating TensorFlow dataset for prediction: {e}")
        return

    print("Generating predictions...")
    try:
        predictions = model.predict(tf_dataset).logits
        probabilities = tf.sigmoid(predictions).numpy()
        output = (probabilities > optimal_thresholds).astype(int)
        print("Prediction complete.")
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return

    print("Saving predictions...")
    try:
        original_label_cols = [col for col in df.columns if col.lower() != 'text']

        if output.shape[0] != len(df):
             raise ValueError(f"Prediction output rows ({output.shape[0]}) != input df rows ({len(df)})")

        if output.shape[1] != len(original_label_cols) and len(original_label_cols) > 0 :
            print(f"Warning: Number of predicted labels ({output.shape[1]}) differs from original label columns ({len(original_label_cols)}). Replacing columns based on model's label order.")
            # Drop original label columns if they exist and we are replacing them
            df_text_col = df[['text']] # Keep text column
            df = df_text_col # Start with only text
            # Add new columns based on model's label order and predictions
            for i in range(num_labels):
                 label_name_from_model = model.config.id2label[i]
                 df[label_name_from_model] = output[:, i]
        elif len(original_label_cols) == 0: # No label columns in input, add based on model
            print(f"No label columns in input CSV. Adding {num_labels} predicted columns based on model config.")
            for i in range(num_labels):
                 label_name_from_model = model.config.id2label[i]
                 df[label_name_from_model] = output[:, i]
        else: # Counts match, overwrite directly
             df[original_label_cols] = output


        archive_name = os.path.splitext(os.path.basename(output_zip))[0] + '.csv'
        df.to_csv(output_zip, index=False, compression=dict(
            method='zip', archive_name=archive_name))
        print(f"Predictions saved to {output_zip} (with {archive_name} inside)")
    except Exception as e:
        print(f"Error saving predictions: {e}")
    print("Prediction script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict multi-label text classification.")
    parser.add_argument("command", choices=["train", "predict"], help="Action to perform: 'train' or 'predict'.")
    parser.add_argument("--model_path", default="saved_model_robertabase", help="Directory to save/load the model and thresholds.") # Default changed
    parser.add_argument("--train_path", default="train.csv", help="Path to the training CSV file.")
    parser.add_argument("--dev_path", default="dev.csv", help="Path to the validation/development CSV file.")
    parser.add_argument("--input_path", default="dev.csv", help="Path to the input CSV file for prediction.")
    parser.add_argument("--output_zip", default="submission_robertabase.zip", help="Path to save the prediction output zip file.") # Default changed

    args = parser.parse_args()

    seed_value = 42
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    if args.command == "train":
        train(model_path=args.model_path, train_path=args.train_path, dev_path=args.dev_path)
    elif args.command == "predict":
        predict(model_path=args.model_path, input_path=args.input_path, output_zip=args.output_zip)
    else:
        print(f"Unknown command: {args.command}")
        