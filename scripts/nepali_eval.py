import os
import re
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clean_nepali_text(text):
    if not isinstance(text, str):
        text = str(text)  # Safely handle non-string inputs (NaN, floats, etc.)

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\u0900-\u097F\d\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class nepaliDataset(Dataset):
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


def evaluate_model_on_nepali(model_path, tokenizer_path, nepali_df, model_label, sentiment_classes=2):
    print(f"\n========== Evaluating {model_label} Model ==========")
    print(f"[INFO] sentiment_classes={sentiment_classes}")

    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded from {tokenizer_path}")

    multi_task_model = MultiTaskXLMR(model_name='xlm-roberta-base', num_sentiment_classes=sentiment_classes)

    multi_task_model.load_state_dict(torch.load(model_path, map_location=device))
    multi_task_model.to(device)
    multi_task_model.eval()
    print(f"Model loaded from {model_path}")

    dataset = nepaliDataset(nepali_df, tokenizer, max_len=128)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    predictions = predict_multitask(multi_task_model, data_loader, device)

    hate_map = {0: 0, 1: 1}
    nepali_df['hate_prediction'] = [hate_map[x] for x in predictions['hate']]

    print("\nSample Predictions (first 5 rows):")
    for idx in range(min(5, len(nepali_df))):
        text_sample = nepali_df['text'].iloc[idx]
        true_label = nepali_df['label'].iloc[idx]   # 'yes'/'no'
        pred_label = nepali_df['hate_prediction'].iloc[idx]
        print(f"Text: {text_sample}")
        print(f"True Label: {true_label}")
        print(f"Hate Prediction: {pred_label}")
        print("-"*50)

    if 'label' in nepali_df.columns:
        print("\nClassification Report (Hate Task):")
        print(classification_report(nepali_df['label'], nepali_df['hate_prediction'], zero_division=0))


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

    nepali_DATA_PATH = './data/nepali/ss_ac_at_txt_unbal.csv'
    if not os.path.exists(nepali_DATA_PATH):
        raise FileNotFoundError(f"nepali dataset not found at {nepali_DATA_PATH}.")

    nepali_df = pd.read_csv(nepali_DATA_PATH, header=None)
    print("Original nepali DataFrame head:\n", nepali_df.head())
    
    
    
    nepali_df.columns = ['label', 'id', 'i', 'text']
    nepali_df.drop(columns=['id', 'i'], inplace=True)
    nepali_df['text'] = nepali_df['text'].astype(str)
    nepali_df['clean_text'] = nepali_df['text'].apply(clean_nepali_text)
    nepali_df['text'] = nepali_df['clean_text']

    for model_label, path_dict in model_paths.items():
        df_copy = nepali_df.copy()
        evaluate_model_on_nepali(
            model_path=path_dict["model"],
            tokenizer_path=path_dict["tokenizer"],
            nepali_df=df_copy,
            model_label=model_label,
            sentiment_classes=path_dict["sentiment_classes"]
        )

if __name__ == "__main__":
    main()
