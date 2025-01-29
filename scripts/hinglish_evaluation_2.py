import os
import re
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clean_text_generic(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\u0900-\u097F\sa-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class HinglishDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()  # 'yes'/'no'
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
            'text': text,
            'label': label
        }


class MultiTaskXLMR(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', num_sentiment_classes=2):
        super(MultiTaskXLMR, self).__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifiers = nn.ModuleDict({
            'hate': nn.Linear(hidden_size, 2),
            'offensive': nn.Linear(hidden_size, 2),
            # param based on English or Marathi:
            'sentiment': nn.Linear(hidden_size, num_sentiment_classes)
        })

        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, task):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifiers[task](cls_output)
        return logits


def predict_multitask(model, data_loader, device):
    model.eval()
    predictions = {'hate': [], 'offensive': [], 'sentiment': []}

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Hate Task
            hate_logits = model(input_ids=input_ids, attention_mask=attention_mask, task='hate')
            hate_preds = torch.argmax(hate_logits, dim=1).cpu().numpy()
            predictions['hate'].extend(hate_preds)

            # Offensive Task
            offn_logits = model(input_ids=input_ids, attention_mask=attention_mask, task='offensive')
            offn_preds = torch.argmax(offn_logits, dim=1).cpu().numpy()
            predictions['offensive'].extend(offn_preds)

            # Sentiment Task
            sent_logits = model(input_ids=input_ids, attention_mask=attention_mask, task='sentiment')
            sent_preds = torch.argmax(sent_logits, dim=1).cpu().numpy()
            predictions['sentiment'].extend(sent_preds)

    return predictions

def evaluate_model_on_hinglish(model_path, tokenizer_path, hinglish_df, model_label, sentiment_classes=2):
    print(f"\n========== Evaluating {model_label} Model ==========")
    print(f"[INFO] sentiment_classes={sentiment_classes}")

    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded from {tokenizer_path}")

    multi_task_model = MultiTaskXLMR(model_name='xlm-roberta-base', num_sentiment_classes=sentiment_classes)

    multi_task_model.load_state_dict(torch.load(model_path, map_location=device))
    multi_task_model.to(device)
    multi_task_model.eval()
    print(f"Model loaded from {model_path}")

    dataset = HinglishDataset(hinglish_df, tokenizer, max_len=128)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    predictions = predict_multitask(multi_task_model, data_loader, device)

    hate_map = {0: 'no', 1: 'yes'}
    hinglish_df['hate_prediction'] = [hate_map[x] for x in predictions['hate']]

    print("\nSample Predictions (first 5 rows):")
    for idx in range(min(5, len(hinglish_df))):
        text_sample = hinglish_df['text'].iloc[idx]
        true_label = hinglish_df['label'].iloc[idx]   # 'yes'/'no'
        pred_label = hinglish_df['hate_prediction'].iloc[idx]
        print(f"Text: {text_sample}")
        print(f"True Label: {true_label}")
        print(f"Hate Prediction: {pred_label}")
        print("-"*50)

    if 'label' in hinglish_df.columns:
        print("\nClassification Report (Hate Task):")
        print(classification_report(hinglish_df['label'], hinglish_df['hate_prediction'], zero_division=0))


def main():
    model_paths = {
        "English": {
            "model": "./models/multi_task_english/multi_task_xlm_roberta.pth",
            "tokenizer": "./models/multi_task_english/tokenizer_xlm_roberta",
            "sentiment_classes": 2
        },
        "Marathi": {
            "model": "./models/marathi/multi_task_xlm_roberta.pth",
            "tokenizer": "./models/marathi/tokenizer_xlm_roberta",
            "sentiment_classes": 3
        }
    }

    HINGLISH_DATA_PATH = './data/hinglish/hate_speech.tsv'
    if not os.path.exists(HINGLISH_DATA_PATH):
        raise FileNotFoundError(f"Hinglish dataset not found at {HINGLISH_DATA_PATH}.")

    hinglish_df = pd.read_csv(HINGLISH_DATA_PATH, sep='\t')
    print("Original Hinglish DataFrame head:\n", hinglish_df.head())

    hinglish_df['text'] = hinglish_df['text'].astype(str)
    hinglish_df['clean_text'] = hinglish_df['text'].apply(clean_text_generic)
    hinglish_df['text'] = hinglish_df['clean_text']

    for model_label, path_dict in model_paths.items():
        df_copy = hinglish_df.copy()
        evaluate_model_on_hinglish(
            model_path=path_dict["model"],
            tokenizer_path=path_dict["tokenizer"],
            hinglish_df=df_copy,
            model_label=model_label,
            sentiment_classes=path_dict["sentiment_classes"]
        )

if __name__ == "__main__":
    main()
