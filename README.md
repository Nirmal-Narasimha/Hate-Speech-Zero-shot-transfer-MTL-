# Cross-Lingual Hate Speech Detection Using Zero-Shot & Multi-Task Learning

## 📌 Overview
This project focuses on **hate speech detection for low-resource languages** using **Zero-Shot Learning (ZSL) and Multi-Task Learning (MTL)**. We leverage **XLM-RoBERTa** and multilingual datasets to classify hate speech across **five languages**:  
**Hindi, Marathi, Bangla, English, and German**.

## 🔍 Methodology
- **Zero-Shot Learning (ZSL):** Enables classification of hate speech in **unseen languages** without direct supervision.
- **Multi-Task Learning (MTL):** Incorporates **sentiment analysis and offensive language detection** as auxiliary tasks.
- **Fine-tuning Strategy:** Improved generalization by training on **linguistically similar languages (e.g., Marathi)**.

## 📊 Results
- **Macro F1-score:** Achieved **0.79** for Hindi.
- **Performance Gain:** **9% improvement** over single-task setups.
- **Evaluation Metrics:** Macro F1-score, Accuracy, Cross-Entropy Loss.

## ⚙️ Technologies Used
- **Model:** XLM-RoBERTa (Cross-Lingual Transformer)
- **Languages Covered:** Hindi, Marathi, Bangla, English, German
- **Tech Stack:** Python, PyTorch, Hugging Face Transformers, NLP Libraries

