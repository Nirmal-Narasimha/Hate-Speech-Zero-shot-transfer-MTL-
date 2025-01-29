import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import os
from transformers import XLMRobertaTokenizer
from sklearn.metrics import classification_report
from torch import nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def clean_text_generic(text):
    text = str(text).lower()  
    text = re.sub(r'http\S+|www\S+', '', text)     
    text = re.sub(r'<.*?>', '', text)              
    text = re.sub(r'[^\u0900-\u097F\sa-zA-Z]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class InferenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()   # "yes"/"no"
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
    def __init__(self, model_name, num_labels_hate, num_labels_offensive, num_labels_sentiment):
        super(MultiTaskXLMR, self).__init__()
        from transformers import XLMRobertaModel
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier_hate = nn.Linear(hidden_size, num_labels_hate)
        self.classifier_offensive = nn.Linear(hidden_size, num_labels_offensive)
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


def evaluate_model_on_nepali(model_path, tokenizer_path, nepali_df, model_label):
    print(f"\n========== Evaluating {model_label} Model ==========")

    # 1. Load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded from {tokenizer_path}")

    num_labels_hate = 2
    num_labels_offensive = 2
    num_labels_sentiment = 3

    multi_task_model = MultiTaskXLMR(
        model_name='xlm-roberta-base',
        num_labels_hate=num_labels_hate,
        num_labels_offensive=num_labels_offensive,
        num_labels_sentiment=num_labels_sentiment
    )

    multi_task_model.load_state_dict(torch.load(model_path, map_location=device))
    multi_task_model.to(device)
    multi_task_model.eval()
    print(f"Model loaded from {model_path}")

    inference_dataset = InferenceDataset(nepali_df, tokenizer, max_len=128)
    data_loader = DataLoader(inference_dataset, batch_size=32, shuffle=False, num_workers=2)

    predictions = predict_multitask(multi_task_model, data_loader, device)

    hate_map = {0: 0, 1: 1}
    nepali_df['hate_prediction'] = [hate_map[x] for x in predictions['hate']]

    print("\nSample Predictions (first 5 rows):")
    for idx in range(min(5, len(nepali_df))):
        text_sample = nepali_df['text'].iloc[idx]
        true_label = nepali_df['label'].iloc[idx]
        pred_label = nepali_df['hate_prediction'].iloc[idx]
        print(f"Text: {text_sample}")
        print(f"True Label: {true_label}")
        print(f"Hate Prediction: {pred_label}")
        print("-"*50)

    if 'label' in nepali_df.columns:
        print("\nClassification Report (Hate Task):")
        print(classification_report(nepali_df['label'], nepali_df['hate_prediction'], zero_division=0))

def main():
    # Paths to all 6 models/tokenizers
    # Adjust to match your environment
    model_paths = {
        "German": {
            "model": "./models/german/multi_task_xlm_roberta.pth",
            "tokenizer": "./models/german/tokenizer_xlm_roberta"
        },
        # "English": {
        #     "model": "./models/multi_task_english/multi_task_xlm_roberta.pth",
        #     "tokenizer": "./models/multi_task_english/tokenizer_xlm_roberta"
        # },
        "English_German": {
            "model": "./models/english_german/multi_task_xlm_roberta.pth",
            "tokenizer": "./models/english_german/tokenizer_xlm_roberta"
        },
        "Bangla": {
            "model": "./models/bangla/multi_task_xlm_roberta.pth",
            "tokenizer": "./models/bangla/tokenizer_xlm_roberta"
        },
        # "Marathi": {
        #     "model": "./models/marathi/multi_task_xlm_roberta.pth",
        #     "tokenizer": "./models/marathi/tokenizer_xlm_roberta"
        # },
        "Marathi_Bangla": {
            "model": "./models/marathi_bangla/multi_task_xlm_roberta.pth",
            "tokenizer": "./models/marathi_bangla/tokenizer_xlm_roberta"
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
    nepali_df['clean_text'] = nepali_df['text'].apply(clean_text_generic)
    nepali_df['text'] = nepali_df['clean_text']

    for model_label, path_dict in model_paths.items():
        df_copy = nepali_df.copy()
        evaluate_model_on_nepali(
            model_path=path_dict["model"],
            tokenizer_path=path_dict["tokenizer"],
            nepali_df=df_copy,
            model_label=model_label
        )

if __name__ == "__main__":
    main()