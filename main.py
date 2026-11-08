import os
import sounddevice as sd
from scipy.io.wavfile import write
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import whisper

DURATION = 5
SAMPLE_RATE = 16000
OUTPUT_FILENAME = "enregistrement_direct.wav"
MODEL_DIR = "./modele_sauvegarde"

# 🛡️ LISTE DES MOTS-CLÉS CRITIQUES (Urgence 5 garantie à 100%)
MOTS_CLES_URGENCE = [
    "au secours",
    "à l'aide",
    "au feu",
    "au meurtre",
    "sauvez-moi",
    "attaque",
    "laissez-moi",
]


def enregistrer_micro(filename, duration, fs):
  print(
      f"\n🎙️ [ENREGISTREMENT] Parlez maintenant ! (Écoute pendant {duration}"
      " secondes...)"
  )
  audio_data = sd.rec(
      int(duration * fs), samplerate=fs, channels=1, dtype="int16"
  )
  sd.wait()
  write(filename, fs, audio_data)
  print("✅ Enregistrement terminé.")


def lancer_pipeline_live():
  print("=" * 60)
  print("🚀 Lancement du module SAFE-AI (Mode Hybride : Règles + IA)")
  print("=" * 60)

  # 1. Enregistrement
  enregistrer_micro(OUTPUT_FILENAME, DURATION, SAMPLE_RATE)

  # 2. Transcription Whisper
  print("\n[1/3] Transcription de l'audio avec Whisper...")
  whisper_model = whisper.load_model("small")
  resultat_whisper = whisper_model.transcribe(OUTPUT_FILENAME, language="fr")
  texte_transcrit = resultat_whisper["text"].strip()
  print(f' 📝 Texte transcrit : "{texte_transcrit}"')

  # 3. ANALYSE HYBRIDE (Sécurité Mots-clés + Modèle IA)
  print("\n[2/3] Analyse du texte (Vérification des mots-clés + Modèle)...")

  texte_minuscule = texte_transcrit.lower()
  urgence_detectee_par_regle = False

  # Étape A : Vérification des mots-clés critiques absolus
  for mot in MOTS_CLES_URGENCE:
    if mot in texte_minuscule:
      score_urgence = 5
      confiance = 1.00  # Certitude absolue par règle de sécurité
      urgence_detectee_par_regle = True
      methode_analyse = "Règle de sécurité critique (Mot-clé détecté)"
      break

  # Étape B : Si aucun mot-clé critique n'est présent, on laisse l'IA (CamemBERT) analyser
  if not urgence_detectee_par_regle:
    if not os.path.exists(MODEL_DIR):
      print(
          "❌ Erreur : Le modèle entraîné n'existe pas. Lancez d'abord"
          " 'python train.py'."
      )
      return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    inputs = tokenizer(
        texte_transcrit, return_tensors="pt", padding=True, truncation=True
    )
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    score_urgence = predicted_class_idx + 1
    confiance = probabilities[0][predicted_class_idx].item()
    methode_analyse = "Modèle CamemBERT (IA)"

  print("\n" + "=" * 60)
  print(" 📊 RÉSULTAT FINAL DE VOTRE MODULE ")
  print("=" * 60)
  print(f"Texte analysé : {texte_transcrit}")
  print(f"🚨 Niveau d'urgence estimé (1 à 5) : ** {score_urgence} / 5 **")
  print(f"Confiance : {confiance * 100:.2f}%")
  print(f"Méthode : {methode_analyse}")
  print("=" * 60)


if __name__ == "__main__":
  lancer_pipeline_live()