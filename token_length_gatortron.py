import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.utils import resample, shuffle
import matplotlib.pyplot as plt
from datasets import concatenate_datasets

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

X_train, y_train, X_test_val, y_test_val = iterative_train_test_split(X,
                                                              y,
                                                              test_size=0.2)

X_test, y_test, X_val, y_val = iterative_train_test_split(X_test_val,
                                                          y_test_val,
                                                          test_size=0.5)

# reconstruct DataFrame after splitting
def reconstruct_df(X, y, X_columns, label_columns):
    df = pd.DataFrame(X, columns=X_columns)
    for i, col in enumerate(label_columns):
        df[col] = y[:, i]
    return df

train_df = reconstruct_df(X_train, y_train, X_columns, label_columns)
test_df = reconstruct_df(X_test, y_test, X_columns, label_columns)
val_df = reconstruct_df(X_val, y_val, X_columns, label_columns)

# group data together
complications_df = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'test': Dataset.from_pandas(test_df),
    'valid': Dataset.from_pandas(val_df)})

# define model and tokenizer
model_name = "UFNLP/gatortron-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def count_tokens(batch):
    outputs = tokenizer(batch["text"], 
                        add_special_tokens=True, 
                        padding=False, # want to see the true length
                        truncation=False) # want to see the true length
    return {"token_count": [len(ids) for ids in outputs['input_ids']]}

# apply function 
tokenized_dataset = complications_df.map(count_tokens, 
                                 batched=True)


# combine all datasets
combined_dataset = concatenate_datasets([
    tokenized_dataset['train'], 
    tokenized_dataset['test'], 
    tokenized_dataset['valid']
])

# get all token counts
all_token_counts = np.array(combined_dataset['token_count'])

# calculate statistics
max_length = 512
num_truncated = np.sum(all_token_counts > max_length)
percent_truncated = (num_truncated / len(all_token_counts)) * 100

print(f"Total number of texts: {len(all_token_counts)}")
print(f"Number of texts that will be truncated: {num_truncated}")
print(f"Percentage of texts that will be truncated: {percent_truncated:.2f}%")
print(f"Mean token count: {np.mean(all_token_counts):.2f}")
print(f"Median token count: {np.median(all_token_counts):.2f}")
print(f"Min token count: {np.min(all_token_counts)}")
print(f"Max token count: {np.max(all_token_counts)}")

# Calculate percentage of texts at various length thresholds
thresholds = [512, 4096]
for threshold in thresholds:
    percent_above = (np.sum(all_token_counts > threshold) / len(all_token_counts)) * 100
    print(f"Percentage of texts with more than {threshold} tokens: {percent_above:.2f}%")

# Plot distribution
mean_tokens = round(np.mean(all_token_counts))
median_tokens = round(np.median(all_token_counts))


plt.figure(figsize=(12, 6))
plt.hist(all_token_counts, bins=5000, edgecolor='black')
plt.title('Distribution of token counts gatortron-base')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.axvline(x=512, color='r', linestyle='--', label='BERT-base')
plt.axvline(x=4096, color='g', linestyle='--', label='Longformer')
plt.xlim(0, 10000)
plt.axvline(x=mean_tokens, color='orange', linestyle='--', label='Mean')
plt.axvline(x=median_tokens, color='purple', linestyle='--', label='Median')

plt.text(mean_tokens, plt.ylim()[1], f'Mean: {int(mean_tokens)}', 
         rotation=90, va='top', ha='right', color='orange')
plt.text(median_tokens, plt.ylim()[1], f'Median: {int(median_tokens)}', 
         rotation=90, va='top', ha='right', color='purple')

plt.legend()

plt.tight_layout()
plt.savefig('combined_token_distribution_gatortron.png', dpi=300)
plt.close()



