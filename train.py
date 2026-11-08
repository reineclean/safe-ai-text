import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

print("🚀 Démarrage de l'entraînement du modèle Safe-AI...")

# 1. Charger le dataset CSV
df = pd.read_csv("dataset_urgence.csv")

# IMPORTANT : Convertir les scores de 1-5 en indices de 0 à 4
# Et renommer la colonne en 'labels' (exigé par Hugging Face)
df["labels"] = df["label"] - 1

# Transformer en dataset Hugging Face
dataset = Dataset.from_pandas(df)

# 2. Charger le tokenizer et le modèle CamemBERT
model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Fonction de tokenisation
def tokenize_function(examples):
  return tokenizer(
      examples["texte"], padding="max_length", truncation=True, max_length=128
  )


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Configuration du modèle avec 5 classes (indices 0, 1, 2, 3, 4)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=5
)

# 3. Paramètres d'entraînement
training_args = TrainingArguments(
    output_dir="./resultats_entrainement",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=1,
    report_to="none",  # Désactive les logs externes superflus
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# 4. Lancer l'entraînement
print("⏳ Entraînement en cours...")
trainer.train()

# 5. Sauvegarder le modèle entraîné
model.save_pretrained("./modele_sauvegarde")
tokenizer.save_pretrained("./modele_sauvegarde")
print("✅ Entraînement terminé ! Modèle sauvegardé dans './modele_sauvegarde'.")