import pandas as pd
import numpy as np
import torch
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.utils import shuffle
from transformers import AutoTokenizer, MegatronBertForSequenceClassification
from datasets import Dataset
from transformers import Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import f1_score, average_precision_score

# load data
df = pd.read_csv("/data/scratch/hhz049/1-diabetes-complications/complications_10year_df.csv", index_col=0)

df['text'] = df['text'].str.lower()

# convert data for splitting 
patid = 'patid'
text = 'text'
label_columns = ['label_nephro', 'label_retino', 'label_neuro']

# shuffle data
df1 = shuffle(df, random_state=42)  # Set random_state for reproducibility

# create y as a 2D numpy array of labels
y = df1[label_columns].values

# create X as a 2D numpy array of patid and text
X = df1[[patid, text]].values

# column names for later use
X_columns = [patid, text]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

X_train, y_train, X_test_val, y_test_val = iterative_train_test_split(X,
                                                              y,
                                                              test_size=0.2)

X_test, y_test, X_val, y_val = iterative_train_test_split(X_test_val,
                                                          y_test_val,
                                                          test_size=0.5)

print(len(X_train))
print(len(X_test))
print(len(X_val))

# reconstruct DataFrame after splitting
def reconstruct_df(X, y, X_columns, label_columns):
    df = pd.DataFrame(X, columns=X_columns)
    for i, col in enumerate(label_columns):
        df[col] = y[:, i]
    return df

train_df = reconstruct_df(X_train, y_train, X_columns, label_columns)
test_df = reconstruct_df(X_test, y_test, X_columns, label_columns)
val_df = reconstruct_df(X_val, y_val, X_columns, label_columns)

# load model (this is your fine tuned model)
model_path = "/data/scratch/hhz049/10_year"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = MegatronBertForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# tokenise dataset
labels = [label for label in test_df.columns if label not in ['patid', 'text']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
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

# make a huggingface dataset
test_dataset = Dataset.from_pandas(test_df)

encoded_test_dataset = test_dataset.map(preprocess_data, 
                                        batched=True, 
                                        remove_columns=test_dataset.column_names)

# set metrics
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # predictions are already sigmoid outputs
    y_pred = (predictions >= threshold).astype(int)
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    auprc = average_precision_score(y_true, predictions, average='micro')
    metrics = {'f1': f1_micro_average,
               'auprc': auprc}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

# define training arguments
training_args = TrainingArguments(
    output_dir="/data/scratch/hhz049/10_year",
    per_device_eval_batch_size=8,
    report_to="none",
)

# create a Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ealuate the model on the test dataset
eval_results = trainer.evaluate(eval_dataset=encoded_test_dataset)
print("Evaluation results:", eval_results)

# dfine bootstrap_metrics function
def bootstrap_metrics(y_true, y_pred, raw_predictions, n_iterations=1000): # bootstrapping 1000 iterations, adjust depending on data size
    results = []
    n_samples = len(y_true)
    
    for _ in range(n_iterations):
        # sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        raw_pred_sample = raw_predictions[indices]
        
        # calculate metrics for this sample (only calculating f1 and AUPRC here, adjust as needed)
        f1_micro = f1_score(y_true=y_true_sample, y_pred=y_pred_sample, average='micro')
        auprc = average_precision_score(y_true_sample, raw_pred_sample, average='micro')
        
        results.append({'f1': f1_micro, 'auprc': auprc})
    
    return results

# get predictions
predictions = trainer.predict(encoded_test_dataset)

y_true = predictions.label_ids
raw_predictions = predictions.predictions
y_pred = (raw_predictions > 0.5).astype(int)

# apply bootstrapping
bootstrap_results = bootstrap_metrics(y_true, y_pred, raw_predictions, n_iterations=1000)

# calculate and print confidence intervals
for metric in bootstrap_results[0].keys():
    values = [result[metric] for result in bootstrap_results]
    ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
    print(f"{metric}: 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]")