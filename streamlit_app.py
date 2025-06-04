# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16, MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import io

# Configuración de la app
st.set_page_config(page_title="Clasificación de Beans", layout="wide")

# --- FUNCIONES ---
def decode_image(byte_data, target_size=(32, 32)):
    image = Image.open(io.BytesIO(byte_data)).convert("RGB")
    image = image.resize(target_size)
    return np.array(image) / 255.0

def prepare_dataset(df, target_size=(32, 32)):
    X = np.array([decode_image(b['bytes'], target_size) for b in df['image']])
    y = to_categorical(df['labels'])
    return X, y

def build_transfer_model(base_model, num_classes):
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test, name):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    st.subheader(f"{name} Classification Report")
    st.text(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

# --- CARGA DE DATOS ---
st.title("Beans Classification")

splits = {
    "train":       "data/train-00000-of-00001.parquet",
    "validation":  "data/validation-00000-of-00001.parquet",
    "test":        "data/test-00000-of-00001.parquet",
}

try:
    df_train = pd.read_parquet("hf://datasets/AI-Lab-Makerere/beans/" + splits["train"])
    df_val = pd.read_parquet("hf://datasets/AI-Lab-Makerere/beans/" + splits["validation"])
    df_test = pd.read_parquet("hf://datasets/AI-Lab-Makerere/beans/" + splits["test"])

    X_train, y_train = prepare_dataset(df_train)
    X_val, y_val = prepare_dataset(df_val)
    X_test, y_test = prepare_dataset(df_test)
    num_classes = y_train.shape[1]

    st.success("¡Datasets cargados correctamente!")
except Exception as e:
    st.error(f"No se pudo cargar el dataset: {e}")
    st.stop()


# --- SELECCIÓN DE MODELO ---
st.sidebar.title("⚙️ Selecciona el modelo a entrenar:")
model_choice = st.sidebar.selectbox(
    "Modelo",
    ("Baseline CNN", "VGG16 Transfer", "MobileNetV2 Transfer")
)

if model_choice == "Baseline CNN":
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
elif model_choice == "VGG16 Transfer":
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    model = build_transfer_model(base_model, num_classes)
elif model_choice == "MobileNetV2 Transfer":
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    model = build_transfer_model(base_model, num_classes)
else:
    st.error("Modelo no reconocido.")
    st.stop()

# --- ENTRENAMIENTO ---
if st.button("Entrenar Modelo"):
    with st.spinner("Entrenando modelo..."):
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(f"{model_choice.replace(' ', '_').lower()}_model.keras", save_best_only=True)
        ]
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=64,
            callbacks=callbacks,
            verbose=0
        )
        st.success(f"¡{model_choice} entrenado con éxito!")

    # --- EVALUACIÓN ---
    evaluate_model(model, X_test, y_test, model_choice)

# --- PREDICCIÓN DE IMAGEN INDIVIDUAL ---
st.subheader("📸 Realiza una predicción con tu propia imagen")

uploaded_file = st.file_uploader("Carga una imagen en formato JPG o PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesar la imagen
    image_resized = image.resize((32, 32))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # forma (1, 32, 32, 3)

    # Realizar la predicción
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Mapea la etiqueta si las tienes definidas
    label_names = ['bean_rust', 'angular_leaf_spot', 'healthy']
    st.success(f"🌟 Predicción: {label_names[predicted_class]} con confianza {confidence:.2f}")


#.\tfenv2\Scripts\activate
#streamlit run streamlit_app.py
