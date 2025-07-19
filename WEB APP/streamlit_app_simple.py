import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import os

# Configurazione della pagina
st.set_page_config(
    page_title="Analisi EEG e Predizione Emozioni",
    page_icon="🧠",
    layout="wide"
)

# Titolo principale
st.title("🧠 Analisi EEG e Predizione Emozioni")
st.markdown("Carica un file CSV con dati EEG per ottenere predizioni emozionali e visualizzazioni")

# Sidebar per informazioni
st.sidebar.header("ℹ️ Informazioni")
st.sidebar.markdown("""
Questa applicazione analizza i dati EEG e predice le emozioni utilizzando:
- **Modello di Machine Learning** per la classificazione
- **Logica Fuzzy** per l'intensità emotiva
- **Visualizzazioni** dei risultati
- **Arte Generativa** basata sulle emozioni

**Nota:** Questa è una versione demo che simula le predizioni.
""")

def create_emotion_probability_plot(probabilities, classes):
    """Crea un grafico a barre delle probabilità delle emozioni"""
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(classes, probabilities, color='lightblue')

    # Evidenzia la barra con probabilità più alta
    bars[np.argmax(probabilities)].set_color('salmon')

    # Aggiungi i valori sulle barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')

    ax.set_title('Probabilità Emozioni Predette dal Modello', pad=20)
    ax.set_xlabel('Emozioni', labelpad=10)
    ax.set_ylabel('Probabilità', labelpad=10)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig

def create_fuzzy_membership_plot(fuzzy_value):
    """Crea un grafico che mostra le curve di appartenenza fuzzy"""
    # Universo dell'output fuzzy
    x = np.arange(0, 1.01, 0.01)

    # Funzioni di appartenenza triangolari semplici
    basso = np.maximum(0, np.minimum(1, (0.5 - x) / 0.5))
    medio = np.maximum(0, np.minimum((x - 0.2) / 0.3, (0.8 - x) / 0.3))
    alto = np.maximum(0, np.minimum((x - 0.5) / 0.5, 1))

    # Creazione grafico
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Disegna le curve di appartenenza
    ax.plot(x, basso, label='Basso', linewidth=2)
    ax.plot(x, medio, label='Medio', linewidth=2)
    ax.plot(x, alto, label='Alto', linewidth=2)

    # Linea verticale rossa tratteggiata che mostra il valore fuzzy calcolato
    ax.axvline(fuzzy_value, color='red', linestyle='--', linewidth=2, label=f'Valore: {fuzzy_value:.2f}')

    # Impostazioni grafiche
    ax.set_title('Appartenenza Fuzzy dello Stato Emotivo', pad=15)
    ax.set_xlabel('Output Normalizzato', labelpad=10)
    ax.set_ylabel('Grado di Appartenenza', labelpad=10)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    return fig

def generate_simple_art(emotion, intensity, width=512, height=512):
    """Genera arte semplice basata sull'emozione"""
    import random
    from PIL import ImageDraw
    
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    
    # Parametri basati sull'emozione
    if emotion == 'POSITIVE':
        colors = [(255, 200, 50), (255, 150, 100), (200, 255, 100)]
        num_shapes = int(50 + 30 * intensity)
    elif emotion == 'NEGATIVE':
        colors = [(100, 50, 50), (150, 100, 100), (80, 80, 120)]
        num_shapes = int(30 + 20 * intensity)
    else:  # NEUTRAL
        colors = [(150, 150, 200), (200, 200, 150), (180, 180, 180)]
        num_shapes = int(40 + 20 * intensity)
    
    # Disegna forme casuali
    for _ in range(num_shapes):
        x1, y1 = random.randint(0, width-50), random.randint(0, height-50)
        x2, y2 = x1 + random.randint(20, 80), y1 + random.randint(20, 80)
        color = random.choice(colors)
        
        shape_type = random.choice(['circle', 'rectangle'])
        if shape_type == 'circle':
            draw.ellipse([x1, y1, x2, y2], fill=color)
        else:
            draw.rectangle([x1, y1, x2, y2], fill=color)
    
    return image

def simulate_prediction(data):
    """Simula una predizione basata sui dati EEG"""
    # Genera probabilità casuali ma realistiche
    np.random.seed(42)  # Per risultati riproducibili
    
    # Simula probabilità basate su alcune statistiche dei dati
    mean_val = np.mean(data.values) if len(data.values) > 0 else 0.5
    
    if mean_val > 0.6:
        probs = [0.2, 0.3, 0.5]  # Più positivo
    elif mean_val < 0.4:
        probs = [0.5, 0.3, 0.2]  # Più negativo
    else:
        probs = [0.3, 0.4, 0.3]  # Neutro
    
    # Aggiungi un po' di casualità
    noise = np.random.normal(0, 0.1, 3)
    probs = np.array(probs) + noise
    probs = np.abs(probs)  # Assicura valori positivi
    probs = probs / np.sum(probs)  # Normalizza
    
    return probs

# Mostra messaggio informativo
st.info("🔧 **Versione Demo**: Questa versione simula le predizioni per dimostrare l'interfaccia Streamlit")

# Upload del file
st.header("📁 Carica File EEG")
uploaded_file = st.file_uploader(
    "Seleziona un file CSV con dati EEG",
    type=['csv'],
    help="Il file deve contenere dati EEG in formato CSV"
)

if uploaded_file is not None:
    try:
        # Leggi il file CSV
        eeg_data = pd.read_csv(uploaded_file)
        
        st.success(f"File caricato con successo! Dimensioni: {eeg_data.shape}")
        
        # Mostra anteprima dei dati
        with st.expander("👀 Anteprima Dati EEG"):
            st.dataframe(eeg_data.head())
            st.write(f"**Colonne:** {list(eeg_data.columns)}")
        
        # Simula la predizione
        with st.spinner("🔄 Elaborazione in corso..."):
            probabilities = simulate_prediction(eeg_data)
            
            # Classi del modello
            model_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
            
            # Trova l'emozione dominante
            dominant_idx = np.argmax(probabilities)
            dominant_emotion = model_classes[dominant_idx]
            dominant_intensity = probabilities[dominant_idx]
            
            # Simula intensità fuzzy
            fuzzy_intensities = {
                'NEGATIVE': float(probabilities[0]),
                'NEUTRAL': float(probabilities[1]),
                'POSITIVE': float(probabilities[2])
            }
        
        # Layout a colonne per i risultati
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🎯 Risultati Predizione")
            
            # Risultati del modello
            st.subheader("Modello di Machine Learning")
            st.metric("Emozione Dominante", dominant_emotion)
            
            # Mostra le probabilità
            prob_df = pd.DataFrame({
                'Emozione': model_classes,
                'Probabilità': probabilities
            })
            st.dataframe(prob_df, use_container_width=True)
            
            # Grafico delle probabilità
            prob_fig = create_emotion_probability_plot(probabilities, model_classes)
            st.pyplot(prob_fig)
            
        with col2:
            st.header("🌊 Analisi Fuzzy")
            
            st.subheader("Intensità Fuzzy")
            st.metric("Intensità Dominante", f"{dominant_intensity:.3f}")
            
            # Mostra le intensità fuzzy
            fuzzy_df = pd.DataFrame({
                'Emozione': list(fuzzy_intensities.keys()),
                'Intensità': list(fuzzy_intensities.values())
            })
            st.dataframe(fuzzy_df, use_container_width=True)
            
            # Grafico fuzzy
            fuzzy_fig = create_fuzzy_membership_plot(dominant_intensity)
            st.pyplot(fuzzy_fig)
        
        # Feature Importance (simulata)
        st.header("📊 Importanza delle Feature")
        if len(eeg_data.columns) > 0:
            # Simula feature importance
            feature_importance = np.random.rand(len(eeg_data.columns))
            most_important_idx = np.argmax(feature_importance)
            most_important_feature = eeg_data.columns[most_important_idx]
            
            st.metric("Feature più Importante", most_important_feature)
            
            # Mostra top 10 feature
            top_n = min(10, len(eeg_data.columns))
            top_indices = np.argsort(feature_importance)[-top_n:][::-1]
            
            importance_df = pd.DataFrame({
                'Feature': [eeg_data.columns[i] for i in top_indices],
                'Importanza': feature_importance[top_indices]
            })
            
            st.bar_chart(importance_df.set_index('Feature'))
        
        # Arte Generativa
        st.header("🎨 Arte Generativa")
        generated_image = generate_simple_art(dominant_emotion, dominant_intensity)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(generated_image, caption=f"Arte basata su: {dominant_emotion}", use_container_width=True)
            
            # Pulsante per scaricare l'immagine
            img_buffer = io.BytesIO()
            generated_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label="💾 Scarica Arte Generata",
                data=img_buffer.getvalue(),
                file_name=f"arte_emotiva_{dominant_emotion.lower()}.png",
                mime="image/png"
            )
            
    except Exception as e:
        st.error(f"Errore durante l'elaborazione del file: {e}")

# Footer
st.markdown("---")
st.markdown("🧠 **Analisi EEG e Predizione Emozioni** - Powered by Streamlit")

# Esempio di dati per test
st.sidebar.markdown("---")
st.sidebar.header("📝 Dati di Test")
if st.sidebar.button("Genera CSV di Esempio"):
    # Crea dati di esempio
    np.random.seed(42)
    sample_data = pd.DataFrame({
        f'EEG_Channel_{i+1}': np.random.rand(100) for i in range(10)
    })
    
    csv_buffer = io.StringIO()
    sample_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    st.sidebar.download_button(
        label="📥 Scarica CSV di Esempio",
        data=csv_buffer.getvalue(),
        file_name="eeg_sample_data.csv",
        mime="text/csv"
    )

