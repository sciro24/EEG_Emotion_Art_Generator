import numpy as np
import random
from PIL import Image, ImageDraw
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf

# --- Definizione variabili linguistiche e universi ---
prob_negative = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_negative')
prob_neutral = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_neutral')
prob_positive = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_positive')
stato_emotivo_fuzzy = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'stato_emotivo_fuzzy')

# --- Funzioni di appartenenza ---
for var in [prob_negative, prob_neutral, prob_positive]:
    var['bassa'] = fuzz.trimf(var.universe, [0, 0, 0.5])
    var['media'] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
    var['alta'] = fuzz.trimf(var.universe, [0.5, 1, 1])

stato_emotivo_fuzzy['basso'] = fuzz.trimf(stato_emotivo_fuzzy.universe, [0, 0, 0.5])
stato_emotivo_fuzzy['medio'] = fuzz.trimf(stato_emotivo_fuzzy.universe, [0.2, 0.5, 0.8])
stato_emotivo_fuzzy['alto'] = fuzz.trimf(stato_emotivo_fuzzy.universe, [0.5, 1, 1])

# --- Regole fuzzy ---
rules = [
    ctrl.Rule(prob_positive['alta'] & prob_negative['bassa'], stato_emotivo_fuzzy['alto']),
    ctrl.Rule(prob_negative['alta'] & prob_positive['bassa'], stato_emotivo_fuzzy['basso']),
    ctrl.Rule(prob_neutral['alta'], stato_emotivo_fuzzy['medio']),
    ctrl.Rule((prob_positive['media'] | prob_neutral['media']) & prob_negative['media'], stato_emotivo_fuzzy['medio']),
    ctrl.Rule(prob_negative['bassa'] & prob_neutral['bassa'] & prob_positive['bassa'], stato_emotivo_fuzzy['basso'])
]

# --- Sistema di controllo e simulazione ---
emotion_ctrl_system = ctrl.ControlSystem(rules)
emotion_simulation = ctrl.ControlSystemSimulation(emotion_ctrl_system)

def get_fuzzy_emotion_state(probabilities, emotion_classes):
    mapping = {'NEGATIVE': 'prob_negative', 'NEUTRAL': 'prob_neutral', 'POSITIVE': 'prob_positive'}
    for emo in emotion_classes:
        if emo in mapping:
            emotion_simulation.input[mapping[emo]] = probabilities[emotion_classes.index(emo)]
    try:
        emotion_simulation.compute()
        fuzzy_output = emotion_simulation.output['stato_emotivo_fuzzy']
        dominant_emotion_idx = np.argmax(probabilities)
        return emotion_classes[dominant_emotion_idx], fuzzy_output
    except Exception as e:
        print(f"Errore calcolo fuzzy: {e}\nInput forniti: {emotion_simulation.input}")
        return None, None

def generate_abstract_art(emotion_label, fuzzy_value, width=512, height=512):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    params = {
        'POSITIVE': {'color': (int(255*fuzzy_value), int(200+50*fuzzy_value), int(50+100*fuzzy_value)),
                     'num_shapes': int(70+50*fuzzy_value), 'size_range': (20,120), 'line_width': int(2+2*fuzzy_value), 'shape': 'polygon'},
        'NEUTRAL': {'color': (int(150+100*fuzzy_value), int(200+50*fuzzy_value), int(255*fuzzy_value)),
                    'num_shapes': int(20+30*fuzzy_value), 'size_range': (30,150), 'line_width': int(1+3*fuzzy_value), 'shape': 'circle'},
        'NEGATIVE': {'color': (int(150+100*(1-fuzzy_value)), int(50+100*(1-fuzzy_value)), int(50+100*(1-fuzzy_value))),
                     'num_shapes': int(80+70*(1-fuzzy_value)), 'size_range': (5,80), 'line_width': int(3+5*(1-fuzzy_value)), 'shape': 'line'}
    }
    p = params.get(emotion_label, {'color': (150,150,150), 'num_shapes':50, 'size_range':(10,100), 'line_width':1, 'shape':'random'})
    for _ in range(p['num_shapes']):
        x1, y1 = random.randint(0,width), random.randint(0,height)
        size = random.randint(*p['size_range'])
        x2, y2 = x1+size, y1+size
        base_color = p['color']
        color = tuple(max(0, min(255, base_color[i] + random.randint(-50,50))) for i in range(3))
        shape_type = p['shape'] if p['shape'] != 'random' else random.choice(['circle', 'rectangle', 'line', 'polygon'])
        if shape_type == 'circle':
            draw.ellipse([x1,y1,x2,y2], fill=color, outline=color, width=p['line_width'])
        elif shape_type == 'rectangle':
            draw.rectangle([x1,y1,x2,y2], fill=color, outline=color, width=p['line_width'])
        elif shape_type == 'line':
            x3, y3 = random.randint(0,width), random.randint(0,height)
            draw.line([x1,y1,x3,y3], fill=color, width=p['line_width'])
        elif shape_type == 'polygon':
            points = [(random.randint(0,width), random.randint(0,height)) for _ in range(random.randint(3,6))]
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
