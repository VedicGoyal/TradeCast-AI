from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

save_dir = "./finbert_local"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"✅ FinBERT downloaded and saved to {save_dir}")
