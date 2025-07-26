import numpy as np
import random
from PIL import Image, ImageDraw
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf

# --- Definizione variabili linguistiche e universi ---

# Input: probabilità predette dal modello (range 0-1)
prob_negative = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_negative')
prob_neutral  = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_neutral')
prob_positive = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_positive')

# Output: intensità fuzzy per ogni emozione (range 0-1)
intensita_negativa = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'intensita_negativa')
intensita_neutra   = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'intensita_neutra')
intensita_positiva = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'intensita_positiva')

# --- Funzioni di appartenenza per input ---
for var in [prob_negative, prob_neutral, prob_positive]:
    var['bassa'] = fuzz.trimf(var.universe, [0, 0, 0.5])
    var['media'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
    var['alta']  = fuzz.trimf(var.universe, [0.5, 1, 1])

# --- Funzioni di appartenenza per output (intensità) ---
for out_var in [intensita_negativa, intensita_neutra, intensita_positiva]:
    out_var['bassa'] = fuzz.trimf(out_var.universe, [0, 0, 0.5])
    out_var['media'] = fuzz.trimf(out_var.universe, [0.3, 0.5, 0.7])
    out_var['alta']  = fuzz.trimf(out_var.universe, [0.5, 1, 1])

# --- Regole fuzzy per intensità negativa ---
rule_neg_1 = ctrl.Rule(prob_negative['alta'], intensita_negativa['alta'])
rule_neg_2 = ctrl.Rule(prob_negative['media'], intensita_negativa['media'])
rule_neg_3 = ctrl.Rule(prob_negative['bassa'], intensita_negativa['bassa'])

# --- Regole fuzzy per intensità neutra ---
rule_neu_1 = ctrl.Rule(prob_neutral['alta'], intensita_neutra['alta'])
rule_neu_2 = ctrl.Rule(prob_neutral['media'], intensita_neutra['media'])
rule_neu_3 = ctrl.Rule(prob_neutral['bassa'], intensita_neutra['bassa'])

# --- Regole fuzzy per intensità positiva ---
rule_pos_1 = ctrl.Rule(prob_positive['alta'], intensita_positiva['alta'])
rule_pos_2 = ctrl.Rule(prob_positive['media'], intensita_positiva['media'])
rule_pos_3 = ctrl.Rule(prob_positive['bassa'], intensita_positiva['bassa'])

# --- Sistema di controllo e simulazione ---
emotion_ctrl_system = ctrl.ControlSystem([
    rule_neg_1, rule_neg_2, rule_neg_3,
    rule_neu_1, rule_neu_2, rule_neu_3,
    rule_pos_1, rule_pos_2, rule_pos_3,
])

emotion_simulation = ctrl.ControlSystemSimulation(emotion_ctrl_system)

def get_fuzzy_emotion_intensity(probabilities, emotion_classes):
    emotion_simulation = ctrl.ControlSystemSimulation(emotion_ctrl_system)  # ← Spostato qui

    # Mappa le probabilità ai rispettivi input fuzzy
    mapping = {'NEGATIVE': 'prob_negative', 'NEUTRAL': 'prob_neutral', 'POSITIVE': 'prob_positive'}
    for emo in emotion_classes:
        if emo in mapping:
            emotion_simulation.input[mapping[emo]] = probabilities[emotion_classes.index(emo)]

    try:
        emotion_simulation.compute()
        # Estrazione output fuzzy
        intensities = {
            'NEGATIVE': emotion_simulation.output['intensita_negativa'],
            'NEUTRAL': emotion_simulation.output['intensita_neutra'],
            'POSITIVE': emotion_simulation.output['intensita_positiva'],
        }
        # Scelta emozione dominante sulla base dell'intensità fuzzy
        dominant_emotion = max(intensities, key=intensities.get)
        dominant_intensity = intensities[dominant_emotion]
        return dominant_emotion, dominant_intensity, intensities
    except Exception as e:
        print(f"Errore calcolo fuzzy: {e}\nInput forniti: {emotion_simulation.input}")
        return None, None, None


def generate_abstract_art(emotion_label, fuzzy_intensities, width=512, height=512):
    # Prendo il valore fuzzy per l'emozione indicata, default 0.5 se non trovato
    fuzzy_value = fuzzy_intensities.get(emotion_label, 0.5)
    
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    
    params = {
        'POSITIVE': {
            'color': (int(255*fuzzy_value), int(200+50*fuzzy_value), int(50+100*fuzzy_value)),
            'num_shapes': int(70+50*fuzzy_value),
            'size_range': (20,120),
            'line_width': int(2+2*fuzzy_value),
            'shape': 'polygon'
        },
        'NEUTRAL': {
            'color': (int(150+100*fuzzy_value), int(200+50*fuzzy_value), int(255*fuzzy_value)),
            'num_shapes': int(20+30*fuzzy_value),
            'size_range': (30,150),
            'line_width': int(1+3*fuzzy_value),
            'shape': 'circle'
        },
        'NEGATIVE': {
            'color': (int(150+100*(1-fuzzy_value)), int(50+100*(1-fuzzy_value)), int(50+100*(1-fuzzy_value))),
            'num_shapes': int(80+70*(1-fuzzy_value)),
            'size_range': (5,80),
            'line_width': int(3+5*(1-fuzzy_value)),
            'shape': 'line'
        }
    }
    
    p = params.get(emotion_label, {'color': (150,150,150), 'num_shapes':50, 'size_range':(10,100), 'line_width':1, 'shape':'random'})
    
    for _ in range(p['num_shapes']):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        size = random.randint(*p['size_range'])
        x2, y2 = x1 + size, y1 + size
        base_color = p['color']
        color = tuple(max(0, min(255, base_color[i] + random.randint(-50,50))) for i in range(3))
        
        shape_type = p['shape'] if p['shape'] != 'random' else random.choice(['circle', 'rectangle', 'line', 'polygon'])
        
        if shape_type == 'circle':
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=color, width=p['line_width'])
        elif shape_type == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=color, width=p['line_width'])
        elif shape_type == 'line':
            x3, y3 = random.randint(0, width), random.randint(0, height)
            draw.line([x1, y1, x3, y3], fill=color, width=p['line_width'])
        elif shape_type == 'polygon':
            points = [(random.randint(0, width), random.randint(0, height)) for _ in range(random.randint(3,6))]
            draw.polygon(points, fill=color, outline=color)
    
    return image

def compute_feature_importance_ffnn(model, input_data_np, class_idx):
    input_tensor = tf.convert_to_tensor(input_data_np, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        preds = model(input_tensor)
        class_output = preds[0, class_idx]
    gradients = tape.gradient(class_output, input_tensor)
    return np.abs(gradients.numpy())[0]
