from datasets import load_dataset, Dataset, DatasetDict
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModel, AutoConfig, MegatronBertForSequenceClassification
from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import EarlyStoppingCallback
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, confusion_matrix, roc_auc_score, multilabel_confusion_matrix, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from skmultilearn.model_selection import iterative_train_test_split

# load your data, this should include both data ['text'] and label columns
df = pd.read_csv("/data/scratch/hhz049/1-diabetes-complications/complications_10year_df.csv", index_col=0)

# rename and clean
df['text'] = df['text'].str.lower()

# convert data for splitting 
patid = 'patid'
text = 'text'
label_columns = ['label_nephro', 'label_retino', 'label_neuro']

# shuffle data
df1 = shuffle(df, random_state=42)

# create y as a 2D numpy array of labels
y = df1[label_columns].values

# create X as a 2D numpy array of patid and text
X = df1[[patid, text]].values

# column names for later use
X_columns = [patid, text]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# use iterative train test split as this is a multi class problem and we want to ensure a balance of all classes in splits
X_train, y_train, X_test_val, y_test_val = iterative_train_test_split(X,
                                                                      y,
                                                                      test_size=0.2)

X_test, y_test, X_val, y_val = iterative_train_test_split(X_test_val,
                                                          y_test_val,
                                                          test_size=0.5)

print(len("Length of X train:", X_train))
print(len("Length of X test:",X_test))
print(len("Length of X val:",X_val))

# reconstruct DataFrame after splitting
def reconstruct_df(X, y, X_columns, label_columns):
    df = pd.DataFrame(X, columns=X_columns)
    for i, col in enumerate(label_columns):
        df[col] = y[:, i]
    return df

train_df = reconstruct_df(X_train, y_train, X_columns, label_columns)
test_df = reconstruct_df(X_test, y_test, X_columns, label_columns)
val_df = reconstruct_df(X_val, y_val, X_columns, label_columns)

# calculate class weights for BCE loss using training data
def calculate_class_weights(y_train):
    num_samples = len(y_train)
    num_classes = y_train.shape[1]
    class_weights = []
    for i in range(num_classes):
        class_count = np.sum(y_train[:, i])
        weight = (num_samples - class_count) / class_count
        class_weights.append(weight)
    return torch.tensor(class_weights, dtype=torch.float32)

class_weights = calculate_class_weights(train_df[label_columns].values)

# group together
complications_df = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'test': Dataset.from_pandas(test_df),
    'valid': Dataset.from_pandas(val_df)})

print(complications_df)

# check number of labels in each split
print("Training dataset nephro...", Counter(train_df['label_nephro']))
print("Test dataset nephro...", Counter(test_df['label_nephro']))
print("Validation dataset nephro...", Counter(val_df['label_nephro']))

print("Training dataset neuro...", Counter(train_df['label_neuro']))
print("Test dataset neuro...", Counter(test_df['label_neuro']))
print("Validation dataset neuro...", Counter(val_df['label_neuro']))

print("Training dataset retino...", Counter(train_df['label_retino']))
print("Test dataset retino...", Counter(test_df['label_retino']))
print("Validation dataset retino...", Counter(val_df['label_retino']))

# create labels column
labels = [label for label in complications_df['train'].features.keys() if label not in ['patid', 'text']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# define model and tokenizer
model_name = "UFNLP/gatortron-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_len=512
# tokenizer.truncation_side='left' # uncomment this line if you want to truncate from left

model = MegatronBertForSequenceClassification.from_pretrained(model_name,
                                                              problem_type="multi_label_classification",
                                                              num_labels=len(labels), # the output layer with have this number of output neurons
                                                              id2label=id2label, 
                                                              label2id=label2id)

# tokenise dataset
def preprocess_data(examples):
  text = examples["text"]
  encoding = tokenizer(text, 
                       padding="max_length", 
                       truncation=True, 
                       max_length=512)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]
    
  encoding["labels"] = labels_matrix.tolist() # tensor of floats is needed here 
  return encoding

encoded_dataset = complications_df.map(preprocess_data, 
                                       batched=True, 
                                       remove_columns=complications_df['train'].column_names)

# set format 
encoded_dataset.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# modify the CustomTrainer to use weighted BCE loss
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            positive_weights = self.class_weights.unsqueeze(0).expand_as(labels)
            negative_weights = torch.ones_like(positive_weights)
            weights = torch.where(labels == 1, positive_weights, negative_weights)
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights, reduction='mean')
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        
        return (loss, outputs) if return_outputs else loss

# define Trainer (adjust this based on the size of your data)
args = TrainingArguments(
    output_dir="/data/scratch/hhz049/10_year", # your output directory
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8, 
    evaluation_strategy="steps",
    eval_steps=5000, 
    save_strategy="steps",
    save_steps=5000, 
    logging_strategy="steps",
    logging_steps=1000, 
    fp16=True,
    max_steps=48000,  # you will need to adjust this based on the size of your data
    seed=42,
    learning_rate=2e-5, 
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="linear",
    gradient_accumulation_steps=1
)

# move model and data to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# set metrics
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # predictions are already sigmoid outputs
    y_pred = (predictions >= threshold).astype(int)
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    precision_samples = precision_score(y_true, y_pred, average='samples', zero_division=0)
    roc_auc = roc_auc_score(y_true, predictions, average='micro')
    auprc = average_precision_score(y_true, predictions, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'precision': precision_samples,
               'roc_auc': roc_auc,
               'accuracy': accuracy,
               'auprc': auprc}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

# Update the trainer
trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["valid"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)], # adjust for different early stopping
    class_weights=class_weights  # pass the class weights
)

# fine tune pre-trained model
trainer.train()

# training and evaluation loss from log history
loss_values = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
eval_loss_values = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]

# apply to test set
eval_results = trainer.evaluate(eval_dataset=encoded_dataset["test"])
print("Evaluation results...", eval_results)

# Get predictions for the test set
predictions = trainer.predict(encoded_dataset["test"])
y_pred = predictions.predictions
y_true = predictions.label_ids

# sigmoid and threshold for binary predictions
y_pred = (y_pred > 0.5).astype(int)

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_columns, zero_division=0))

# Confusion matrix
print("Confusion Matrix:")
print(multilabel_confusion_matrix(y_true, y_pred))

# save model
trainer.save_model("/data/scratch/hhz049/10_year")
tokenizer.save_pretrained("/data/scratch/hhz049/10_year")

