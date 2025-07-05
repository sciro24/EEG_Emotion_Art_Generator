# 🎨 EEG Emotion to Art Generator

## Riconoscimento delle Emozioni da Segnali EEG e Generazione Artistica Astratta

Questo progetto presenta un sistema integrato che utilizza segnali cerebrali (EEG) per riconoscere le emozioni dell'utente e generare opere d'arte astratte in base allo stato emotivo rilevato. L’obiettivo è creare un ponte tra il mondo interno delle emozioni umane e l’espressione artistica digitale attraverso tecnologie avanzate di deep learning e logica fuzzy.

---

### 🧠 Progetto

Il sistema si compone di due fasi principali:

1. **Classificazione delle Emozioni da EEG**:  
   - Utilizzo di una rete neurale feedforward per classificare i segnali EEG in tre categorie: **Negativo**, **Neutro**, **Positivo**.
   - Addestramento su un dataset di segnali EEG con pre-elaborazione (standardizzazione) e valutazione tramite accuratezza e altre metriche.

2. **Generazione Artistica Astratta**:  
   - Inferenza dell’emozione dominante e raffinamento tramite **logica fuzzy** per catturare sfumature e intensità emotive.
   - Creazione di arte astratta ispirata dallo stato emotivo rilevato.

---

### 📁 File Principali

- `EEG_Emotion_Classifier.ipynb`: Notebook per l'addestramento del modello di classificazione EEG.
- `EEG_Emotion_Art_Generator.ipynb`: Notebook per l’inferenza e la generazione artistica.
- `emotion_classifier_model.keras`: Modello Keras addestrato per la classificazione delle emozioni.
- `scaler.pkl`: Oggetto StandardScaler usato per la standardizzazione dei dati.
- `support_function.py`: Funzioni ausiliarie, inclusa la logica fuzzy per il raffinamento emotivo.
- `eeg_brainwave_dataset.csv`: Dataset utilizzato per l’addestramento.
- `eeg_sample_negative.csv` (e simili): Campioni EEG per testare il processo di inferenza.

---

### ⚙️ Tecnologie Utilizzate

- **Python** come linguaggio principale
- **Google Colab** per lo sviluppo e testing
- **Keras / TensorFlow** per il deep learning
- **Scikit-learn** per preprocessing e metriche
- **NumPy & Pandas** per manipolazione dei dati
- **StandardScaler** per la normalizzazione dei segnali EEG
- **Logica Fuzzy** per migliorare l’interpretazione dello stato emotivo

---

### 📊 Metriche Valutate

- **Accuracy**: Percentuale di emozioni correttamente riconosciute nel test set

---

### 👤 Autori

- **Diego Scirocco 558658** – Ideatore e sviluppatore del progetto

---

### 🔄 Futuri Sviluppi

- Integrazione con dispositivi EEG in tempo reale
- Utilizzo di GAN o modelli a diffusione per arte più sofisticata
- Estensione a emozioni più specifiche e personalizzabili
- Interfaccia utente interattiva per esperienza completa

---

> 💡 *“L’intelligenza artificiale incontra l’anima umana nell’arte.”*
