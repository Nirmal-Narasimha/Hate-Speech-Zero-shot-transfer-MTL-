import random
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import os

DATA_DIR = './data/english/'      
HINDI_DATA_DIR = './data/hindi/'  
MODEL_DIR = './models/english_german/'
os.makedirs(MODEL_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def balance_dataset(df, target_labels, samples_per_class):
    balanced_df = pd.DataFrame()
    for label in target_labels:
        label_df = df[df['label'] == label]
        available_samples = len(label_df)
        if available_samples < samples_per_class:
            raise ValueError(f"Not enough samples for label '{label}'. Required: {samples_per_class}, Available: {available_samples}")
        sampled_df = label_df.sample(n=samples_per_class, random_state=42)
        balanced_df = pd.concat([balanced_df, sampled_df], ignore_index=True)
    return balanced_df
    
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_german(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text


german_hs_dataset_1 = pd.read_csv(f'./data/german/germanrefugees.csv', header=None, sep='\t')
german_hs_dataset_2 = pd.read_csv(f'./data/german/polly_withhate_dez16-20.csv', sep='\t')
german_hs_dataset_3 = pd.read_csv(f'./data/german/hasoc.csv', header=None, sep='\t')
german_hs_dataset_4 = pd.read_csv(f'./data/german/bretschneider.csv', sep='\t')
german_hs_dataset_5 = pd.read_csv(f'./data/german/germeval2018.csv', header=None, sep='\t', encoding="utf-16")
german_hs_dataset_6 = pd.read_csv(f'./data/german/hatr.csv', header=None, sep='\t')
german_hs_dataset_7 = pd.read_csv(f'./data/german/german_dataset.tsv', sep='\t')
german_hs_dataset_8 = pd.read_csv(f'./data/german/hasoc_de_test_gold.tsv', sep='\t')
german_hs_dataset_9 = pd.read_csv(f'./data/german/hate_speech.csv')

german_hs_dataset_1.columns = ['label', 'text']
german_hs_dataset_1 = german_hs_dataset_1.loc[:, ['text', 'label']]
german_hs_dataset_2.columns = ['label', 'text']
german_hs_dataset_2 = german_hs_dataset_1.loc[:, ['text', 'label']]
german_hs_dataset_3.columns = ['label', 'text']
german_hs_dataset_3 = german_hs_dataset_1.loc[:, ['text', 'label']]
german_hs_dataset_4.columns = ['label', 'text']
german_hs_dataset_4 = german_hs_dataset_1.loc[:, ['text', 'label']]
german_hs_dataset_5.columns = ['label', 'text']
german_hs_dataset_5 = german_hs_dataset_1.loc[:, ['text', 'label']]
german_hs_dataset_6.columns = ['label', 'text']
german_hs_dataset_6 = german_hs_dataset_1.loc[:, ['text', 'label']]

# Dropping the unrequired Columns
german_hs_dataset_7.drop(['text_id', 'task_1'], axis=1, inplace=True)
german_hs_dataset_8.drop(['text_id', 'task_1'], axis=1, inplace=True)

# Changing the column name
german_hs_dataset_7.rename(columns={'task_2': 'label'}, inplace=True)
german_hs_dataset_8.rename(columns={'task_2': 'label'}, inplace=True)
german_hs_dataset_7['label'] = german_hs_dataset_7['label'].map({'NONE': 'n', 'HATE': 'hs', 'OFFN': 'p', 'PRFN': 'p'})
german_hs_dataset_8['label'] = german_hs_dataset_8['label'].map({'NONE': 'n', 'HATE': 'hs', 'OFFN': 'p', 'PRFN': 'p'})

german_offn = pd.concat([german_hs_dataset_1,
                      german_hs_dataset_2,
                      german_hs_dataset_3,
                      german_hs_dataset_4,
                      german_hs_dataset_5,
                      german_hs_dataset_6,
                      german_hs_dataset_7,
                      german_hs_dataset_8], axis=0, ignore_index=True)

german_sa = pd.read_csv(f'./data/german/sentiment_analysis.tsv', sep='\t')

german_hate = pd.read_csv(f'./data/german/hate_speech.csv')

# Sentiment Data
german_sa['label'] = german_sa['label'].map({'negative':-1, 'neutral':0, 'positive':1})
print("Sentiment Dataset:")
print(german_sa.head())

# Hate Data
german_hate['label'] = german_hate['label'].map({1:'HATE', 0:'NOT'})
print("Hate Dataset:")
print(german_hate.head())

# Offensive Data
german_offn['label'] = german_offn['label'].map({'p':'OFFN', 'n':'NOT'})
print("Offensive Dataset:")
print(german_offn.head())

german_sa['text'] = german_sa['text'].apply(clean_text_german)
german_hate['text'] = german_hate['text'].apply(clean_text_german)
german_offn['text'] = german_offn['text'].apply(clean_text_german)


hate_dataset_path = os.path.join(DATA_DIR, 'hate_offn_data.csv')  # Update with your actual file path

if not os.path.exists(hate_dataset_path):
    raise FileNotFoundError(f"Hate Speech dataset not found at {hate_dataset_path}. Please download it from https://github.com/t-davidson/hate-speech-and-offensive-language and place it in the specified directory.")

# Load hate speech data
hate_df = pd.read_csv(hate_dataset_path)

hate_df.rename(columns={'tweet': 'text', 'class': 'label'}, inplace=True)
hate_df['text'] = hate_df['text'].apply(clean_text)

print("\nHate Speech Dataset Sample:")
print(hate_df.head())

print("\nHate Speech Label Distribution:")
print(hate_df['label'].value_counts())

hate_target_labels = ['HATE', 'NOT']
offensive_target_labels = ['OFFN', 'NOT']

samples_per_class_hate = 2500
samples_per_class_offensive = 2500

# Balance Hate Speech Dataset for 'HATE' and 'NOT'
balanced_hate_df = balance_dataset(hate_df, hate_target_labels, samples_per_class_hate)

# Balance Offensive Dataset for 'OFFN' and 'NOT'
if 'OFFN' in hate_df['label'].unique():
    balanced_offensive_df = balance_dataset(hate_df[hate_df['label'].isin(['OFFN', 'NOT'])], offensive_target_labels, samples_per_class_offensive)
    print("\nBalanced Offensive Dataset Label Distribution:")
    print(balanced_offensive_df['label'].value_counts())
else:
    balanced_offensive_df = pd.DataFrame(columns=hate_df.columns)
    print("\nNo 'OFFN' labels found in Hate Speech dataset.")

print("\nBalanced Hate Speech Dataset Label Distribution:")
print(balanced_hate_df['label'].value_counts())

sentiment140_path = os.path.join(DATA_DIR, 'sentiment140.csv')  # Update with your actual file path

if not os.path.exists(sentiment140_path):
    raise FileNotFoundError(f"Sentiment140 dataset not found at {sentiment140_path}. Please download it from https://www.kaggle.com/kazanova/sentiment140 and place it in the specified directory.")

# Load Sentiment140 data
sentiment140_df = pd.read_csv(sentiment140_path, encoding='latin-1', header=None)

sentiment140_df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

sentiment_map = {0: -1, 4: 1}
sentiment140_df['mapped_label'] = sentiment140_df['sentiment'].map(sentiment_map)

sentiment140_df['label'] = sentiment140_df['mapped_label']

sentiment140_df['text'] = sentiment140_df['text'].apply(clean_text)

print("\nSentiment140 Dataset Sample:")
print(sentiment140_df.head())

print("\nSentiment140 Label Distribution:")
print(sentiment140_df['mapped_label'].value_counts())

sentiment_target_labels = [-1, 1]

samples_per_class_sentiment = 2500  # 2500 Negative, 2500 Positive

balanced_sentiment140_df = balance_dataset(sentiment140_df, sentiment_target_labels, samples_per_class_sentiment)

print("\nBalanced Sentiment140 Dataset Label Distribution:")
print(balanced_sentiment140_df['mapped_label'].value_counts())



tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
MAX_LEN = 128

class MultiTaskHateDataset(Dataset):
    def __init__(self, hate_data, sentiment_data, offensive_data, tokenizer, max_len):
        self.hate_data = hate_data
        self.sentiment_data = sentiment_data
        self.offensive_data = offensive_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._create_multitask_data()

    def _create_multitask_data(self):
        # Add a task identifier to each dataset
        ds_hate = self.hate_data.copy()
        ds_hate['task'] = 'hate'
        ds_hate = ds_hate.rename(columns={'label': 'task_label'})

        ds_offensive = self.offensive_data.copy()
        ds_offensive['task'] = 'offensive'
        ds_offensive = ds_offensive.rename(columns={'label': 'task_label'})

        ds_sentiment = self.sentiment_data.copy()
        ds_sentiment['task'] = 'sentiment'
        ds_sentiment = ds_sentiment.rename(columns={'label': 'task_label'})

        combined = pd.concat([ds_hate, ds_offensive, ds_sentiment], ignore_index=True)

        return combined

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        task = row['task']
        label = row['task_label']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        if task == 'hate':
            # Labels: 'HATE' -> 1, 'NOT' -> 0
            label_map = {'HATE': 1, 'NOT': 0}
            label = label_map.get(label, 0)

        elif task == 'offensive':
            # Labels: 'OFFN' -> 1, 'NOT' -> 0
            label_map = {'OFFN': 1, 'NOT': 0}
            label = label_map.get(label, 0)

        elif task == 'sentiment':
            # Labels: 'POSITIVE' -> 2, 'NEUTRAL' -> 1, 'NEGATIVE' -> 0
            label_map = {-1: 0, 0: 1, 1: 2}
            label = label_map.get(label, 1)

        else:
            label = -1  # Undefined task

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'task': task,
            'labels': torch.tensor(label, dtype=torch.long)
        }

multitask_dataset = MultiTaskHateDataset(
    hate_data=pd.concat([balanced_hate_df, german_hate[german_hate['label']=='NOT'][:1200], german_hate[german_hate['label']=='HATE'][:1200]]).sample(frac=1),          # Balanced Hate Speech Dataset
    sentiment_data=pd.concat([balanced_sentiment140_df, german_sa[german_sa['label']==-1][:794], german_sa[german_sa['label']==0][:800], german_sa[german_sa['label']==1][:800]]).sample(frac=1),         # Balanced Sentiment Dataset
    offensive_data=pd.concat([balanced_offensive_df, german_offn[german_offn['label']=='NOT'][:1200], german_offn[german_offn['label']=='OFFN'][:1117]]).sample(frac=1), # Balanced Offensive Dataset
    tokenizer=tokenizer,
    max_len=MAX_LEN
)


train_size = 0.8
train_data, test_data = train_test_split(
    multitask_dataset.data,
    test_size=1 - train_size,
    random_state=42,
    stratify=multitask_dataset.data[['task', 'task_label']]
)

train_dataset = MultiTaskHateDataset(
    hate_data=train_data[train_data['task'] == 'hate'],
    sentiment_data=train_data[train_data['task'] == 'sentiment'],
    offensive_data=train_data[train_data['task'] == 'offensive'],
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_dataset = MultiTaskHateDataset(
    hate_data=test_data[test_data['task'] == 'hate'],
    sentiment_data=test_data[test_data['task'] == 'sentiment'],
    offensive_data=test_data[test_data['task'] == 'offensive'],
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

print("\nTraining Set Label Distribution:")
print(train_dataset.data['task'].value_counts())

print("\nValidation Set Label Distribution:")
print(test_dataset.data['task'].value_counts())

BATCH_SIZE = 32
NUM_WORKERS = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

class MultiTaskXLMR(nn.Module):
    def __init__(self, model_name, num_labels_hate, num_labels_offensive, num_labels_sentiment):
        super(MultiTaskXLMR, self).__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Classification head for hate task
        self.classifier_hate = nn.Linear(hidden_size, num_labels_hate)

        # Classification head for offensive task
        self.classifier_offensive = nn.Linear(hidden_size, num_labels_offensive)

        # Classification head for sentiment task
        self.classifier_sentiment = nn.Linear(hidden_size, num_labels_sentiment)

        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, task):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)

        if task == 'hate':
            logits = self.classifier_hate(cls_output)
        elif task == 'offensive':
            logits = self.classifier_offensive(cls_output)
        elif task == 'sentiment':
            logits = self.classifier_sentiment(cls_output)
        else:
            raise ValueError(f"Unknown task: {task}")

        return logits

num_labels_hate = 2          
num_labels_offensive = 2     
num_labels_sentiment = 3     

model = MultiTaskXLMR(
    model_name='xlm-roberta-base',
    num_labels_hate=num_labels_hate,
    num_labels_offensive=num_labels_offensive,
    num_labels_sentiment=num_labels_sentiment
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

labels_hate = train_dataset.hate_data['task_label'].map({'HATE': 1, 'NOT': 0}).values
class_weights_hate = compute_class_weight(class_weight='balanced', classes=np.unique(labels_hate), y=labels_hate)
class_weights_hate = torch.tensor(class_weights_hate, dtype=torch.float).to(device)

labels_offensive = train_dataset.offensive_data['task_label'].map({'OFFN': 1, 'NOT': 0}).values
class_weights_offensive = compute_class_weight(class_weight='balanced', classes=np.unique(labels_offensive), y=labels_offensive)
class_weights_offensive = torch.tensor(class_weights_offensive, dtype=torch.float).to(device)

labels_sentiment = train_dataset.sentiment_data['task_label'].map({-1: 0, 0: 1, 1: 2}).values
class_weights_sentiment = compute_class_weight(class_weight='balanced', classes=np.unique(labels_sentiment), y=labels_sentiment)
class_weights_sentiment = torch.tensor(class_weights_sentiment, dtype=torch.float).to(device)

criterion_hate = nn.CrossEntropyLoss(weight=class_weights_hate)
criterion_offensive = nn.CrossEntropyLoss(weight=class_weights_offensive)
criterion_sentiment = nn.CrossEntropyLoss(weight=class_weights_sentiment)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

EPOCHS = 5
total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% of total steps for warm-up
    num_training_steps=total_steps
)

def train_epoch_multitask(
    model,
    data_loader,
    optimizer,
    device,
    scheduler,
    criterion_hate,
    criterion_offensive,
    criterion_sentiment
):
    model.train()
    losses = []

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tasks = batch['task']
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        unique_tasks = set(tasks)

        total_loss = 0

        for task in unique_tasks:
            indices = [i for i, t in enumerate(tasks) if t == task]
            if not indices:
                continue

            task_input_ids = input_ids[indices]
            task_attention_mask = attention_mask[indices]
            task_labels = labels[indices]

            # Forward pass
            logits = model(input_ids=task_input_ids, attention_mask=task_attention_mask, task=task)

            # Compute loss based on the task
            if task == 'hate':
                loss = criterion_hate(logits, task_labels)
            elif task == 'offensive':
                loss = criterion_offensive(logits, task_labels)
            elif task == 'sentiment':
                loss = criterion_sentiment(logits, task_labels)
            else:
                raise ValueError(f"Unknown task: {task}")

            # Accumulate loss
            total_loss += loss

        # Backward pass and optimization
        total_loss.backward()

        # Gradient Clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Record the loss
        losses.append(total_loss.item())

        if (batch_idx + 1) % 100 == 0:
            print(f'Batch {batch_idx + 1}/{len(data_loader)} - Loss: {total_loss.item():.4f}')

    return np.mean(losses)

def eval_model_multitask(
    model,
    data_loader,
    criterion_hate,
    criterion_offensive,
    criterion_sentiment,
    device
):
    model.eval()
    losses = []
    all_labels = {'hate': [], 'offensive': [], 'sentiment': []}
    all_preds = {'hate': [], 'offensive': [], 'sentiment': []}

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tasks = batch['task']
            labels = batch['labels'].to(device)

            unique_tasks = set(tasks)

            # Initialize total loss for the batch
            total_loss = 0

            for task in unique_tasks:
                indices = [i for i, t in enumerate(tasks) if t == task]
                if not indices:
                    continue

                task_input_ids = input_ids[indices]
                task_attention_mask = attention_mask[indices]
                task_labels = labels[indices]

                # Forward pass
                logits = model(input_ids=task_input_ids, attention_mask=task_attention_mask, task=task)

                # Compute loss based on the task
                if task == 'hate':
                    loss = criterion_hate(logits, task_labels)
                elif task == 'offensive':
                    loss = criterion_offensive(logits, task_labels)
                elif task == 'sentiment':
                    loss = criterion_sentiment(logits, task_labels)
                else:
                    raise ValueError(f"Unknown task: {task}")

                # Accumulate loss
                total_loss += loss

                # Predictions
                preds = torch.argmax(logits, dim=1)

                # Collect labels and predictions
                all_labels[task].extend(task_labels.cpu().numpy())
                all_preds[task].extend(preds.cpu().numpy())

            # Record loss
            losses.append(total_loss.item())

    f1 = {}
    precision = {}
    recall = {}

    for task in all_labels:
        precision[task], recall[task], f1[task], _ = precision_recall_fscore_support(
            all_labels[task], all_preds[task], average='weighted', zero_division=0
        )

    avg_loss = np.mean(losses)

    return avg_loss, precision, recall, f1

def detailed_evaluation(model, data_loader, device):
    model.eval()
    all_labels = {'hate': [], 'offensive': [], 'sentiment': []}
    all_preds = {'hate': [], 'offensive': [], 'sentiment': []}

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tasks = batch['task']
            labels = batch['labels'].to(device)

            unique_tasks = set(tasks)

            for task in unique_tasks:
                indices = [i for i, t in enumerate(tasks) if t == task]
                if not indices:
                    continue

                task_input_ids = input_ids[indices]
                task_attention_mask = attention_mask[indices]
                task_labels = labels[indices]

                logits = model(input_ids=task_input_ids, attention_mask=task_attention_mask, task=task)

                preds = torch.argmax(logits, dim=1)

                all_labels[task].extend(task_labels.cpu().numpy())
                all_preds[task].extend(preds.cpu().numpy())

    for task in all_labels:
        print(f'Classification Report for {task.capitalize()}:')
        if task == 'hate':
            target_names = ['NOT', 'HATE']
        elif task == 'offensive':
            target_names = ['NOT', 'OFFN']
        elif task == 'sentiment':
            target_names = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        else:
            target_names = []
        print(classification_report(all_labels[task], all_preds[task], target_names=target_names, zero_division=0))
        print('-' * 50)

best_f1 = {'hate': 0, 'offensive': 0, 'sentiment': 0}
loss_history = {'train': [], 'val': []}

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    # Training
    train_loss = train_epoch_multitask(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        criterion_hate=criterion_hate,
        criterion_offensive=criterion_offensive,
        criterion_sentiment=criterion_sentiment
    )

    print(f'Train loss: {train_loss:.4f}')
    loss_history['train'].append(train_loss)

    # Evaluation
    val_loss, val_precision, val_recall, val_f1 = eval_model_multitask(
        model=model,
        data_loader=test_loader,
        criterion_hate=criterion_hate,
        criterion_offensive=criterion_offensive,
        criterion_sentiment=criterion_sentiment,
        device=device
    )

    print(f'Validation loss: {val_loss:.4f}')
    print(f'Validation Precision: Hate: {val_precision["hate"]:.4f}, Offensive: {val_precision["offensive"]:.4f}, Sentiment: {val_precision["sentiment"]:.4f}')
    print(f'Validation Recall: Hate: {val_recall["hate"]:.4f}, Offensive: {val_recall["offensive"]:.4f}, Sentiment: {val_recall["sentiment"]:.4f}')
    print(f'Validation F1 Score: Hate: {val_f1["hate"]:.4f}, Offensive: {val_f1["offensive"]:.4f}, Sentiment: {val_f1["sentiment"]:.4f}')
    print()

    loss_history['val'].append(val_loss)

    # Check and save the best model
    for task in best_f1:
        torch.save(model.state_dict(), f'./models/english_german/{epoch}.pth')
        if val_f1[task] > best_f1[task]:
            best_f1[task] = val_f1[task]
            checkpoint_path = f'./models/english_german/best_model_{task}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model for task '{task}' saved to {checkpoint_path}")


model_save_path = './models/english_german/multi_task_xlm_roberta.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model state_dict saved to {model_save_path}")

tokenizer_save_path = './models/english_german/tokenizer_xlm_roberta'
tokenizer.save_pretrained(tokenizer_save_path)
print(f"Tokenizer saved to {tokenizer_save_path}")

optimizer_save_path = './models/english_german/optimizer.pth'
torch.save(optimizer.state_dict(), optimizer_save_path)
print(f"Optimizer state_dict saved to {optimizer_save_path}")

scheduler_save_path = './models/english_german/scheduler.pth'
torch.save(scheduler.state_dict(), scheduler_save_path)
print(f"Scheduler state_dict saved to {scheduler_save_path}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), loss_history['train'], label='Training Loss')
plt.plot(range(1, EPOCHS + 1), loss_history['val'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

tokenizer_load_path = './models/english_german/tokenizer_xlm_roberta'
tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_load_path)
print(f"Tokenizer loaded from {tokenizer_load_path}")

model_load_path = './models/english_german/multi_task_xlm_roberta.pth'
model.load_state_dict(torch.load(model_load_path, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from {model_load_path}")

detailed_evaluation(model, test_loader, device)

model_load_path = './models/english_german/2.pth'
model.load_state_dict(torch.load(model_load_path, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from {model_load_path}")

hindi_tsv_path1 = './data/hindi/hasoc2019_hi_test_gold_2919.tsv'
hindi_tsv_path2 = './data/hindi/hindi_dataset.tsv'

hindi_df1 = pd.read_csv(hindi_tsv_path1, sep='\t')  # Adjust 'sep' if different
hindi_df2 = pd.read_csv(hindi_tsv_path2, sep='\t')  # Adjust 'sep' if different

hindi_combined_df = pd.concat([hindi_df1, hindi_df2], axis=0, ignore_index=True)
hindi_combined_df['label'] = hindi_combined_df['task_1'].map({'NOT': 'NOT', 'HOF': 'HATE'})
print(f"Combined Hindi Dataset Shape: {hindi_combined_df.shape}")
hindi_combined_df['clean_text'] = hindi_combined_df['text'].apply(clean_text)

hindi_combined_df = hindi_combined_df[['text', 'label', 'clean_text']]

class InferenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.texts = dataframe['clean_text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text,  # Optional: to keep track of the original text,
            'label': label
        }

MAX_LEN = 128
inference_dataset = InferenceDataset(
    dataframe=hindi_combined_df,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

BATCH_SIZE = 32
NUM_WORKERS = 32

inference_loader = DataLoader(
    inference_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False
)

def predict_tasks(model, data_loader, device):
    model.eval()
    predictions = {'hate': [], 'offensive': [], 'sentiment': []}

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            for task in ['hate', 'offensive', 'sentiment']:
                logits = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions[task].extend(preds)

    return predictions

predictions = predict_tasks(model, inference_loader, device)

print("\nSample Predictions:")
for i in range(5):
    print(f"Text: {hindi_combined_df['text'].iloc[i]}")
    print(f"True Label: {hindi_combined_df['label'].iloc[i]}")
    print(f"Hate Prediction: {'HATE' if predictions['hate'][i] == 1 else 'NOT'}")
    print(f"Offensive Prediction: {'OFFN' if predictions['offensive'][i] == 1 else 'NOT'}")
    sentiment_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
    print(f"Sentiment Prediction: {sentiment_map.get(predictions['sentiment'][i], 'UNKNOWN')}")
    print("-" * 50)

hate_map = {0: 'NOT', 1: 'HATE'}
hindi_combined_df['hate_prediction'] = predictions['hate']
hindi_combined_df['hate_prediction'] = hindi_combined_df['hate_prediction'].map(hate_map)

print("Classification Report for Hate Task:")
print(classification_report(hindi_combined_df['label'], hindi_combined_df['hate_prediction'], zero_division=0))