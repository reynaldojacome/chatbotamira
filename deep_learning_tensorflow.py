import json
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# Descargar recursos de NLTK necesarios
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_wordnet_pos(word):
    """
    Mapea la etiqueta POS al primer carácter que acepta lemmatize().

    Args:
        word (str): La palabra para la cual se debe encontrar la etiqueta POS.

    Returns:
        str: La etiqueta POS correspondiente de WordNet.
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Lematizador de NLTK
lemmatizer = WordNetLemmatizer()

# Cargar el archivo JSON
with open('conversations.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Inicializar listas para las preguntas y categorías
patterns = []
tags = []
responses = {}

# Procesar las conversaciones
for conversation in data:
    tag = conversation['tag']
    patterns.extend(conversation['patterns'])
    tags.extend([tag] * len(conversation['patterns']))
    responses[tag] = conversation['responses']

# Crear el diccionario de respuestas y guardarlo como conversations_category_answers.pkl
with open('conversations_category_answers.pkl', 'wb') as file:
    pickle.dump(responses, file)

# Guardar el diccionario de respuestas como conversations_category_answers.json
with open('conversations_category_answers.json', 'w', encoding='utf-8') as file:
    json.dump(responses, file, ensure_ascii=False, indent=4)

def preprocess_text(text):
    """
    Preprocesa el texto tokenizándolo, lematizándolo y limpiando caracteres no alfanuméricos.

    Args:
        text (str): Texto a procesar.

    Returns:
        str: Texto preprocesado.
    """
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word)) for word in words]
    words = [re.sub(r'\W+', '', word) for word in words if re.sub(r'\W+', '', word)]
    return ' '.join(words)

# Preprocesar las preguntas
patterns = [preprocess_text(pattern) for pattern in patterns]

# Vectorizar las preguntas y guardarlo como conversations_vectorizer_bow.pkl
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns).toarray()
with open('conversations_vectorizer_bow.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Codificar las categorías y guardarlas como conversations_categories.pkl
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(pd.DataFrame(tags))
categories = encoder.categories_[0].tolist()
with open('conversations_categories.pkl', 'wb') as file:
    pickle.dump(categories, file)

# Construir el modelo de Deep Learning
model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X, y, epochs=200, batch_size=8, verbose=1)

# Guardar el modelo como un archivo .h5
model.save('conversations_model.h5')

def predict_response(text):
    """
    Predice la respuesta a una pregunta dada usando el modelo entrenado.

    Args:
        text (str): La pregunta para la cual se desea predecir la respuesta.

    Returns:
        str: La respuesta predicha.
    """
    text = preprocess_text(text)
    bow = vectorizer.transform([text]).toarray()
    prediction = model.predict(bow)
    tag = encoder.inverse_transform(prediction)[0][0]
    return responses[tag][0]

print(predict_response("Mi signo es Tauro"))
