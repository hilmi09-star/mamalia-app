from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model('mamalia_model_trained.h5')

# Folder untuk menyimpan gambar upload
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ganti dengan class kamu sesuai urutan saat training (train_generator.class_indices)
class_names = ['badger', 'bat', 'bear', 'camel', 'cat', 'chimpanzee', 'cow', 'deer', 'dog', 'dolphin',
               'donkey', 'elephant', 'fox', 'giraffe', 'goat', 'gorilla', 'hamster', 'hare', 'hedgehog', 'hippopotamus',
               'horse', 'hyena', 'kangaroo', 'koala', 'leopard', 'lion', 'llama', 'mole', 'monkey', 'mouse',
               'otter', 'panda', 'pig', 'polar_bear', 'porcupine', 'rabbit', 'raccoon', 'rat', 'reindeer', 'rhinoceros',
               'sheep', 'squirrel', 'tiger', 'weasel', 'zebra']

def predict_image(filepath):
    # Buka gambar dan ubah ke RGB
    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224))  # Harus sesuai dengan ukuran saat training

    # Preprocess: array → scale → expand dim
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions)) * 100  # Dalam %

    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != "":
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            label, confidence = predict_image(filepath)

            message = f"Gambar ini diprediksi sebagai <strong>{label}</strong> dengan keyakinan <strong>{confidence:.2f}%</strong>."

            return render_template("index.html",
                                   label=label,
                                   confidence=confidence,
                                   image=file.filename,
                                   message=message)
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
    app.run(debug=True)

