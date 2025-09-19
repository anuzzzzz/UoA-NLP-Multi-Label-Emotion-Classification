# UoA NLP - Multi-Label Emotion Classification

## Project Overview

- **Institution:** University of Auckland  
- **Course:** Deep Learning / NLP  
- **Performance:** Highest score in class  
- **Task:** Multi-label emotion classification  
- **Model:** RoBERTa-base with advanced threshold tuning  
- **Evaluation Metric:** Micro F1-Score  

## Problem Statement

Given a text, predict the presence of any of the following 7 emotions:
- **admiration**
- **amusement** 
- **gratitude**
- **love**
- **pride**
- **relief**
- **remorse**

This is a **multi-label classification** problem where texts can have multiple emotions simultaneously.

### Example
**Input Text:**
```
Thanks for the reply! I appreciate your input. Please keep me in the loop, I'd love to be more active with this if possible.
```

**Predicted Labels:**
```
admiration, gratitude, love
```

## Technical Approach

### 1. Model Architecture

```python
model_name = "roberta-base"
MAX_LENGTH = 64
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
```

**Key Components:**
- **Base Model:** RoBERTa-base (125M parameters)
- **Classification Head:** Multi-label binary classification
- **Optimizer:** AdamW with warmup and weight decay
- **Loss Function:** Binary Cross-Entropy with logits

### 2. Advanced Training Strategy

```python
# Optimizer with warmup
optimizer, schedule = create_optimizer(
    init_lr=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=int(num_train_steps * 0.1),  # 10% warmup
    weight_decay_rate=0.01
)

# Early stopping with patience
PATIENCE = 3
MAX_EPOCHS = 10
```

**Training Features:**
- **Early Stopping:** Monitors validation F1 with patience
- **Learning Rate Scheduling:** Warmup + linear decay
- **Weight Decay:** L2 regularization (0.01)
- **Manual Training Loop:** Custom implementation for better control

### 3. Per-Label Threshold Optimization

The key innovation: **individual threshold tuning for each emotion class**

```python
# Find optimal threshold for each label
best_thresholds_arr = np.full(num_labels, 0.5)
threshold_candidates = np.arange(0.1, 0.9, 0.01)

for i in range(num_labels):
    # Test each threshold candidate for this label
    for threshold_candidate in threshold_candidates:
        temp_preds = (val_probs > best_thresholds_arr).astype(int)
        temp_preds[:, i] = (val_probs[:, i] > threshold_candidate).astype(int)
        current_f1_micro = f1_score(true_labels, temp_preds, average='micro')
        
        # Update if improvement found
        if current_f1_micro > best_f1_for_this_label_step:
            optimal_threshold_for_this_label_idx = threshold_candidate
```

**Why This Works:**
- Different emotions have different base rates
- Some emotions are harder to detect (need lower thresholds)
- Others are easier to distinguish (can use higher thresholds)
- Micro F1 optimization balances precision/recall across all labels

### 4. Robust Prediction Pipeline

```python
def predict(model_path, input_path, output_zip):
    # Load model and optimal thresholds
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load per-label thresholds
    with open(threshold_file_path, "r") as f:
        threshold_data = json.load(f)
    
    # Apply thresholds for final predictions
    output = (probabilities > optimal_thresholds).astype(int)
```

## Key Technical Features

### 1. Comprehensive Error Handling
- Graceful handling of missing files, data format issues
- Robust tokenization with fallbacks
- Model loading validation

### 2. Flexible Data Processing
```python
# Auto-detect text column
potential_text_cols = {'text', 'comment_text', 'body'}
# Dynamic label detection
label_names = [col for col in columns if col not in text_cols]
```

### 3. Production-Ready Code
- Modular train/predict functions
- Command-line interface
- Proper model serialization
- Threshold persistence

## Performance Analysis

### Why This Approach Succeeded

1. **Strong Base Model:** RoBERTa-base provides robust text understanding
2. **Optimal Thresholds:** Per-label tuning significantly improves F1
3. **Proper Training:** Early stopping prevents overfitting
4. **Multi-label Awareness:** Designed specifically for multi-label tasks

### Technical Innovations

**Per-Label Threshold Optimization:**
- Instead of using 0.5 for all classes
- Finds optimal threshold for each emotion individually  
- Maximizes overall micro F1-score
- Accounts for class imbalance and difficulty differences

**Micro F1 Optimization:**
- Direct optimization of the evaluation metric
- Better than macro F1 for imbalanced datasets
- Focuses on overall classification accuracy

## Data Format

### Training Data Structure
```csv
text,admiration,amusement,gratitude,love,pride,relief,remorse
"My favourite food is anything I didn't have to cook myself.",0,0,0,0,0,0,0
"Thanks for your reply:) until then hubby and I will anxiously wait 😝",0,0,1,0,0,0,0
```

### Submission Format
- ZIP file containing predictions.csv
- Same format as training data
- Binary predictions (0/1) for each emotion

## Usage

### Training
```bash
python nn.py train --model_path saved_model_robertabase --train_path train.csv --dev_path dev.csv
```

### Prediction  
```bash
python nn.py predict --model_path saved_model_robertabase --input_path test.csv --output_zip submission.zip
```

## Technical Stack

- **Framework:** TensorFlow + Transformers
- **Model:** RoBERTa-base (HuggingFace)
- **Data Processing:** pandas, datasets, numpy
- **Evaluation:** sklearn F1-score
- **Languages:** Python

## Repository Structure

```
UoA-NLP-Multi-Label-Emotion-Classification/
├── nn.py                           # Main training/prediction script
├── README.md                       # This file  
├── requirements.txt                # Dependencies
├── saved_model_robertabase/        # Trained model artifacts
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── optimal_thresholds_per_label.json
├── data/                          # Data directory
│   ├── train.csv
│   ├── dev.csv  
│   └── test.csv
└── submissions/
    └── submission_robertabase.zip  # Final predictions
```

## Key Achievements

### 1. Class-Leading Performance
- Achieved highest F1-score in class
- Demonstrates mastery of transformer fine-tuning
- Sophisticated engineering approach

### 2. Advanced Technical Implementation  
- Per-label threshold optimization
- Robust error handling and validation
- Production-ready code architecture

### 3. Multi-Label Classification Expertise
- Understanding of multi-label vs multi-class
- Proper evaluation metric selection (micro F1)
- Class imbalance handling

## Lessons Learned

### What Worked Well
1. **RoBERTa-base** provided strong baseline performance
2. **Per-label thresholds** gave significant F1 improvements  
3. **Early stopping** prevented overfitting effectively
4. **Micro F1 optimization** aligned training with evaluation

### Technical Insights
- Multi-label classification requires different strategies than multi-class
- Threshold tuning is crucial for optimal performance
- Transformer fine-tuning benefits from proper hyperparameter selection
- Production code requires comprehensive error handling

## Future Improvements

### Model Enhancements
- **Larger Models:** RoBERTa-large or DeBERTa
- **Ensemble Methods:** Multiple model averaging
- **Advanced Architectures:** Multi-task learning approaches

### Training Optimizations  
- **Focal Loss:** Handle class imbalance better
- **Learning Rate Scheduling:** More sophisticated schedules
- **Data Augmentation:** Text augmentation techniques

---

*This project demonstrates advanced NLP engineering skills, achieving top performance through sophisticated transformer fine-tuning and optimization techniques.*