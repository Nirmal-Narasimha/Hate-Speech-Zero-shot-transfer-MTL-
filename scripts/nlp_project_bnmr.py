# -*- coding: utf-8 -*-
"""NLP_Project_BnMr.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GpLXXEPvdgslqdHw0KD2WlRtkviRFjBc
"""

from google.colab import drive
drive.mount('/content/drive')

import random
import pandas as pd
import numpy as np
import re
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report

def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

set_seed(42)

# data_dir = '/kaggle/input/nlp-data'
# output_dir = '/kaggle/working'

data_dir = '/content/drive/MyDrive/NLP/data'
output_dir = '/content/drive/MyDrive/NLP'

mr_sent_train = pd.read_csv(f'{data_dir}/marathi/L3Cube-MahaNLP/Sentiment/tweets-train.csv')
mr_sent_test = pd.read_csv(f'{data_dir}/marathi/L3Cube-MahaNLP/Sentiment/tweets-test.csv')
mr_sent_val = pd.read_csv(f'{data_dir}/marathi/L3Cube-MahaNLP/Sentiment/tweets-valid.csv')

mr_sent = pd.concat([mr_sent_train, mr_sent_val, mr_sent_test], axis=0, ignore_index=True)
mr_sent.rename(columns={'tweet': 'text'}, inplace=True)
mr_sent = mr_sent.sample(frac=1)
print("Marathi Sentiment Dataset:")
print(mr_sent.head())



mr_hate_offn_train = pd.read_excel(f'{data_dir}/marathi/L3Cube-MahaNLP/HateOffensive/hate_train.xlsx')
mr_hate_offn_test = pd.read_excel(f'{data_dir}/marathi/L3Cube-MahaNLP/HateOffensive/hate_test.xlsx')
mr_hate_offn_val = pd.read_excel(f'{data_dir}/marathi/L3Cube-MahaNLP/HateOffensive/hate_valid.xlsx')

mr_hate_offn = pd.concat([mr_hate_offn_train, mr_hate_offn_val, mr_hate_offn_test], axis=0, ignore_index=True)
mr_hate_offn = mr_hate_offn.sample(frac=1)
print("\nMarathi and Offensive Speech Dataset:")
print(mr_hate_offn.head())

bn_sent_train = pd.read_csv(f'{data_dir}/bangla/SAIL/BN_data_train.tsv', sep='\t')
bn_sent_test = pd.read_csv(f'{data_dir}/bangla/SAIL/BN_data_test.tsv', sep='\t')
bn_sent_val = pd.read_csv(f'{data_dir}/bangla/SAIL/BN_data_dev.tsv', sep='\t')

bn_sent = pd.concat([bn_sent_train, bn_sent_val, bn_sent_test], axis=0, ignore_index=True)
bn_sent['label'] = bn_sent['class_label'].map({'BN_NEG':-1, 'BN_NEU':0, 'BN_POS':1})
bn_sent = bn_sent.drop(columns=['class_label', 'id'])
bn_sent = bn_sent.sample(frac=1)
print("Bangla Sentiment Dataset:")
print(bn_sent.head())



bn_hate_train = pd.read_csv(f'{data_dir}/bangla/BD-SHS/train.csv')
bn_hate_test = pd.read_csv(f'{data_dir}/bangla/BD-SHS/test.csv')
bn_hate_val = pd.read_csv(f'{data_dir}/bangla/BD-SHS/val.csv')

bn_hate = pd.concat([bn_hate_train, bn_hate_val, bn_hate_test], axis=0, ignore_index=True)
bn_hate['label'] = bn_hate['hate speech'].map({1:'HATE', 0:'NOT'})
bn_hate.rename(columns={'sentence': 'text'}, inplace=True)
bn_hate = bn_hate.drop(columns=['target', 'type', 'hate speech'])
bn_hate = bn_hate.sample(frac=1)
print("\nBangla Hate Dataset:")
print(bn_hate.head())



bn_offn_train = pd.read_json(f'{data_dir}/bangla/HASOC2024/train.json')
bn_offn_test = pd.read_json(f'{data_dir}/bangla/HASOC2024/test.json')

bn_offn = pd.concat([bn_offn_train, bn_offn_test], axis=0, ignore_index=True)
bn_offn['label'] = bn_offn['offensive_gold'].map({'O':'OFFN', 'N':'NOT'})
bn_offn = bn_offn.drop(columns=['code_mixed_gold', 'offensive_gold', 'target_gold'])
bn_offn = bn_offn.sample(frac=1)
print("\nBangla Offensive Dataset:")
print(bn_offn.head())

#TODO - more preprocessing - special characters, emojis

def clean_text(text):
  text = text.lower()
  text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
  text = re.sub(r'\@\w+|\#','', text)
  text = re.sub(r'\s+', ' ', text).strip()
  return text

mr_hate_offn['clean_text'] = mr_hate_offn['text'].apply(clean_text)
mr_sent['clean_text'] = mr_sent['text'].apply(clean_text)

bn_offn['clean_text'] = bn_offn['text'].apply(clean_text)
bn_hate['clean_text'] = bn_hate['text'].apply(clean_text)
bn_sent['clean_text'] = bn_sent['text'].apply(clean_text)

print('Marathi Datasets:')
print(mr_hate_offn['label'].value_counts())
print(mr_sent['label'].value_counts())

print('\nBangla Datasets:')
print(bn_hate['label'].value_counts())
print(bn_offn['label'].value_counts())
print(bn_sent['label'].value_counts())

label_maps = {
  'hate': {'HATE': 1, 'NOT': 0},
  'offensive': {'OFFN': 1, 'NOT': 0},
  'sentiment': {-1: 0, 0: 1, 1: 2}
}

mr_not = mr_hate_offn[mr_hate_offn['label']=='NOT']
mr_hate = mr_hate_offn[mr_hate_offn['label']=='HATE']
mr_offn = mr_hate_offn[mr_hate_offn['label']=='OFFN']
mr_hate_dataset = pd.concat([mr_hate[:3125], mr_not[:3125]], axis=0)
mr_offn_dataset = pd.concat([mr_offn[:3125], mr_not[3125:2*3125]], axis=0)

mr_sent_neg = mr_sent[mr_sent['label']==-1]
mr_sent_neu = mr_sent[mr_sent['label']==0]
mr_sent_pos = mr_sent[mr_sent['label']==1]
mr_sent_dataset = pd.concat([mr_sent_neg[:4000], mr_sent_neu[:4000], mr_sent_pos[:4000]])

bn_hate_dataset = pd.concat([bn_hate[bn_hate['label']=='NOT'][:2750], bn_hate[bn_hate['label']=='HATE'][:2750]])


hate_dataset = pd.concat([mr_hate_dataset, bn_hate_dataset]).sample(frac=1)
sent_dataset = pd.concat([mr_sent_dataset, bn_sent]).sample(frac=1)
offn_dataset = pd.concat([mr_offn_dataset, bn_offn]).sample(frac=1)

print(hate_dataset['label'].value_counts())
print(sent_dataset['label'].value_counts())
print(offn_dataset['label'].value_counts())

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

class MultiTaskDataset(Dataset):
  def __init__(self, hate_data, sentiment_data, offensive_data, label_maps, tokenizer):
    self.hate_data = hate_data
    self.sentiment_data = sentiment_data
    self.offensive_data = offensive_data
    self.label_maps = label_maps

    self.tokenizer = tokenizer
    self.max_len = 128
    self.data = self.combine_data()

  def combine_data(self):
    ds_hate = self.hate_data.copy()
    ds_hate['task'] = 'hate'
    ds_hate = ds_hate.rename(columns={'label': 'task_label'})

    ds_offensive = self.offensive_data.copy()
    ds_offensive['task'] = 'offensive'
    ds_offensive = ds_offensive.rename(columns={'label': 'task_label'})

    ds_sentiment = self.sentiment_data.copy()
    ds_sentiment['task'] = 'sentiment'
    ds_sentiment = ds_sentiment.rename(columns={'label': 'task_label'})

    return pd.concat([ds_hate, ds_offensive, ds_sentiment])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row = self.data.iloc[idx]

    text = row['clean_text']
    task = row['task']
    label = row['task_label']

    encoding = self.tokenizer.encode_plus(text,
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          padding='max_length',
                                          truncation=True,
                                          return_token_type_ids=False,
                                          return_attention_mask=True,
                                          return_tensors='pt')

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'task': task,
        'text': text,
        'labels': torch.tensor(self.label_maps[task][label], dtype=torch.long)
    }

multitask_dataset = MultiTaskDataset(hate_dataset, sent_dataset, offn_dataset, label_maps, tokenizer)
train_data, test_data = train_test_split(multitask_dataset.data, test_size=0.1, random_state=42, stratify=multitask_dataset.data[['task', 'task_label']])

train_dataset = MultiTaskDataset(
  hate_data=train_data[train_data['task']=='hate'],
  sentiment_data=train_data[train_data['task']=='sentiment'],
  offensive_data=train_data[train_data['task']=='offensive'],
  label_maps=label_maps,
  tokenizer=tokenizer
)
print(f"\nTrain Dataset:\n{train_dataset.data['task'].value_counts()}")


test_dataset = MultiTaskDataset(
  hate_data=test_data[test_data['task']=='hate'],
  sentiment_data=test_data[test_data['task']=='sentiment'],
  offensive_data=test_data[test_data['task']=='offensive'],
  label_maps=label_maps,
  tokenizer=tokenizer
)
print(f"\nTest Dataset:\n{test_dataset.data['task'].value_counts()}")

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class MultiTaskModel(nn.Module):
  def __init__(self, num_labels_hate, num_labels_offensive, num_labels_sentiment):
    super(MultiTaskModel, self).__init__()

    self.encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base')
    hidden_size = self.encoder.config.hidden_size

    self.classifier = nn.ModuleDict({
      'hate': nn.Linear(hidden_size, num_labels_hate),
      'offensive': nn.Linear(hidden_size, num_labels_offensive),
      'sentiment': nn.Linear(hidden_size, num_labels_sentiment)
    })

    self.dropout = nn.Dropout(0.3)

  def forward(self, input_ids, attention_mask, task):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls_output = outputs.last_hidden_state[:, 0, :]
    cls_output = self.dropout(cls_output)
    return self.classifier[task](cls_output)

model = MultiTaskModel(len(label_maps['hate'].keys()), len(label_maps['offensive'].keys()), len(label_maps['sentiment'].keys()))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# class_weight = {}
# for task in label_maps.keys():
labels_hate = train_dataset.hate_data['task_label'].map(label_maps['hate']).values
class_weights_hate = compute_class_weight(class_weight='balanced', classes=np.unique(labels_hate), y=labels_hate)
class_weights_hate = torch.tensor(class_weights_hate, dtype=torch.float).to(device)

labels_offensive = train_dataset.offensive_data['task_label'].map(label_maps['offensive']).values
class_weights_offensive = compute_class_weight(class_weight='balanced', classes=np.unique(labels_offensive), y=labels_offensive)
class_weights_offensive = torch.tensor(class_weights_offensive, dtype=torch.float).to(device)

labels_sentiment = train_dataset.sentiment_data['task_label'].map(label_maps['sentiment']).values
class_weights_sentiment = compute_class_weight(class_weight='balanced', classes=np.unique(labels_sentiment), y=labels_sentiment)
class_weights_sentiment = torch.tensor(class_weights_sentiment, dtype=torch.float).to(device)

criterion = {
  'hate': nn.CrossEntropyLoss(weight=class_weights_hate),
  'offensive': nn.CrossEntropyLoss(weight=class_weights_offensive),
  'sentiment': nn.CrossEntropyLoss(weight=class_weights_sentiment)
}

EPOCHS = 5
total_steps = EPOCHS * len(train_loader)
optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=int(0.1*total_steps),
  num_training_steps=total_steps
)

def train_model(model, data_loader, optimizer, device, scheduler, criterion):

  model.train()

  losses = []
  for _, batch in enumerate(data_loader):

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    tasks = batch['task']
    labels = batch['labels'].to(device)

    optimizer.zero_grad()

    total_loss = 0
    for task in set(tasks):
      indices = [i for i, t in enumerate(tasks) if t==task]
      if not indices:
        continue
      logits = model(input_ids=input_ids[indices], attention_mask=attention_mask[indices], task=task)
      total_loss += criterion[task](logits, labels[indices])

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    losses.append(total_loss.item())

    if (_+1)%100==0:
      print(f'For Batch {_+1}/{len(data_loader)}, Loss={total_loss.item()}')

  return np.mean(losses)

def eval_model(model, data_loader, device, criterion):

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

      total_loss = 0
      for task in set(tasks):
        indices = [i for i, t in enumerate(tasks) if t==task]
        if not indices:
          continue
        logits = model(input_ids=input_ids[indices], attention_mask=attention_mask[indices], task=task)
        total_loss += criterion[task](logits, labels[indices])

        all_labels[task].extend(labels[indices].cpu().numpy())
        all_preds[task].extend(torch.argmax(logits, dim=1).cpu().numpy())

      losses.append(total_loss.item())

    report = {'Precision': {}, 'Recall': {}, 'F1': {}}
    for task in all_labels:
        report['Precision'][task], report['Recall'][task], report['F1'][task], _ = precision_recall_fscore_support(all_labels[task], all_preds[task], average='weighted', zero_division=0)

    return np.mean(losses), report

best_score = {'hate': 0, 'offensive': 0, 'sentiment': 0}
loss_history = {'train': [], 'val': []}

for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')

    train_loss = train_model(model, train_loader, optimizer, device, scheduler, criterion)
    loss_history['train'].append(train_loss)
    print(f'Train loss: {train_loss}')

    val_loss, val_report = eval_model(model, test_loader, device, criterion)
    print(f'Validation loss: {val_loss}')
    for score_type in val_report.keys():
      output_str = f'Validation {score_type}: '
      for task in best_score.keys():
        output_str += f'{task.title()}: {val_report[score_type][task]}, '
      print(output_str)

    loss_history['val'].append(val_loss)

    torch.save(model.state_dict(), f'{output_dir}/{epoch+1}.pth')
    for task in best_score.keys():
      if val_report['F1'][task] > best_score[task]:
        best_score[task] = val_report['F1'][task]
        torch.save(model.state_dict(), f'{output_dir}/best_model_{task}.pth')

tokenizer.save_pretrained(f'{output_dir}/tokenizer')
torch.save(optimizer.state_dict(), f'{output_dir}/optimizer.pth')
torch.save(scheduler.state_dict(), f'{output_dir}/scheduler.pth')



# Evaluation

label_revmaps = {
  'hate': {1: 'HATE', 0: 'NOT'},
  'offensive': {1: 'OFFN', 0: 'NOT'},
  'sentiment': {0: -1, 1: 0, 2: 1}
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

task = 'hate'

tokenizer = XLMRobertaTokenizer.from_pretrained(f'{output_dir}/tokenizer')
model.load_state_dict(torch.load(f'{output_dir}/best_model_{task}.pth', map_location=device))
model.to(device)
model.eval()
print('Model and tokenizer loaded')

hi_train = pd.read_csv(f'{data_dir}/hindi/HASOC2019/hindi_dataset.tsv', sep='\t')
hi_test = pd.read_csv(f'{data_dir}/hindi/HASOC2019/hasoc2019_hi_test_gold_2919.tsv', sep='\t')

hi_combined = pd.concat([hi_train, hi_test], axis=0)
hi_combined['label'] = hi_combined['task_1'].map({'NOT': 'NOT', 'HOF': 'HATE'})
hi_combined['clean_text'] = hi_combined['text'].apply(clean_text)

class SingleTaskDataset(Dataset):
  def __init__(self, dataframe, tokenizer):
    self.texts = dataframe['clean_text'].tolist()
    self.labels = dataframe['label'].tolist()
    self.tokenizer = tokenizer
    self.max_len = 128

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = self.texts[idx]
    label = self.labels[idx]

    encoding = self.tokenizer.encode_plus(text,
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          padding='max_length',
                                          truncation=True,
                                          return_token_type_ids=False,
                                          return_attention_mask=True,
                                          return_tensors='pt')

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'text': text,
        'label': label
    }


BATCH_SIZE = 16
inference_dataset = SingleTaskDataset(hi_combined, tokenizer)
inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False)

predictions = []
with torch.no_grad():
  for batch in inference_loader:
    logits = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), task=task)
    predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
hi_combined['hate_prediction'] = predictions
hi_combined['hate_prediction'] = hi_combined['hate_prediction'].map(label_revmaps[task])


print("\nSample Predictions:")
for i in range(5):
  print(f"\nText: {hi_combined['text'].iloc[i]}")
  print(f"Clean Text: {hi_combined['clean_text'].iloc[i]}")
  print(f"True Label: {hi_combined['label'].iloc[i]}")
  print(f"{task.title()} Prediction: {hi_combined['hate_prediction'].iloc[i]}")

print(f"\nClassification Report for {task}:\n{classification_report(hi_combined['label'], hi_combined['hate_prediction'], zero_division=0)}")



"""**References:**
1.   https://www.kaggle.com/code/harshjain123/bert-for-everyone-tutorial-implementation
2. https://www.analyticsvidhya.com/blog/2023/06/step-by-step-bert-implementation-guide/
3. https://discuss.pytorch.org/t/dealing-with-imbalanced-datasets-in-pytorch/22596/5
4. https://medium.com/analytics-vidhya/pre-processing-tweets-for-sentiment-analysis-a74deda9993e
5. https://medium.com/gumgum-tech/an-easy-recipe-for-multi-task-learning-in-pytorch-that-you-can-do-at-home-1e529a8dfb7f
6. https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch

**Datasets:**
1. https://hasocfire.github.io/hasoc/2024/call_for_participation.html
2. https://www.kaggle.com/datasets/naurosromim/bdshs
3. https://github.com/banglanlp/bnlp-resources/tree/main/sentiment
4. https://github.com/l3cube-pune/MarathiNLP/tree/main

"""