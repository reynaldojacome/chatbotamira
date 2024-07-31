import json
import nltk
import string
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Descargar recursos necesarios de nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Inicializar lematizador y stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process_text(text):
    """
    Procesa el texto dado: tokeniza, convierte a minúsculas, elimina stopwords
    y lematiza las palabras.

    Args:
        text (str): Texto a procesar.

    Returns:
        list: Lista de tokens procesados.
    """
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Leer el archivo conversations.json
with open('conversations.json', 'r') as file:
    data = json.load(file)

# Inicializar listas para vocabulario y tags
vocabulario = []
tags = []

# Procesar cada conversación
for conversation in data:
    patterns = conversation['patterns']
    tag = conversation['tag']
    
    # Procesar y agregar palabras de patterns al vocabulario
    for pattern in patterns:
        processed_pattern = process_text(pattern)
        vocabulario.extend(processed_pattern)
    
    # Agregar tag a la lista de tags
    tags.append(tag.lower())

# Eliminar duplicados y ordenar
vocabulario = sorted(set(vocabulario))
tags = sorted(set(tags))

# Guardar vocabulario y tags en archivos pickle
with open('vocabulario.pkl', 'wb') as f:
    pickle.dump(vocabulario, f)

with open('tags.pkl', 'wb') as f:
    pickle.dump(tags, f)

# Crear bolsa de palabras para patterns
vectorizer = CountVectorizer(tokenizer=process_text)
patterns_corpus = [' '.join(process_text(pattern)) for conversation in data for pattern in conversation['patterns']]
X = vectorizer.fit_transform(patterns_corpus)

# Guardar bolsa de palabras en un archivo CSV
df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df_bow.to_csv('bow_amira_patterns.csv', index=False)
