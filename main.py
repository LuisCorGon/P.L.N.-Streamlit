import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Descargar recursos de NLTK si no lo has hecho ya
nltk.download('punkt')
nltk.download('stopwords')

# Carga del dataset
dataset = pd.read_csv("lasolananews.csv")

# Inspecciona las primeras filas
print(dataset.head())

# Definir stopwords en español (puedes adaptar esto según el idioma)
stop_words = set(stopwords.words("spanish"))

# Función para limpiar el texto
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar signos de puntuación
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios extra
    tokens = word_tokenize(text)  # Tokenización
    tokens = [word for word in tokens if word not in stop_words]  # Quitar stopwords
    return ' '.join(tokens)

# Aplicar la limpieza al dataset
dataset['cleaned_text'] = dataset['Text'].apply(clean_text)

# Verificar el resultado
print(dataset[['Text', 'cleaned_text']].head())

# Crear el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=2250)  # Puedes ajustar el número de características

# Ajustar el vectorizador a los textos limpios y transformarlos
X = tfidf_vectorizer.fit_transform(dataset['cleaned_text']).toarray()

# Verificar la forma de la matriz de características
print(X.shape)

# Las etiquetas (Topic) serán nuestras variables objetivo
y = dataset['Topic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar la distribución de los datos
print(f"Conjunto de entrenamiento: {X_train.shape}, Conjunto de prueba: {X_test.shape}")

# Crear una instancia del modelo
model = LogisticRegression(C = 100, max_iter=1000, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Verificar el entrenamiento completado
print("Modelo entrenado con éxito.")

# Realizar predicciones
y_pred = model.predict(X_test)

# Generar un reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Calcular precisión global
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")



def get_model():
    return model

def get_y():
    return y_test, y_pred


