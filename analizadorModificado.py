import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('spam_assassin.csv')
registros_iniciales = len(data)
print()
print('=' * 90)
print('Registros iniciales: ', registros_iniciales)

data = data.drop_duplicates(subset=['text'])
data.reset_index(drop=True, inplace=True)
registros_unicos = len(data)
print('Registros sin duplicados: ', registros_unicos)
print("Registros eliminados: ", registros_iniciales - registros_unicos)

print('=' * 90)

cantidad_spam = data['target'].sum()
print('Correos SPAM: ', cantidad_spam)
cantidad_no_spam = len(data) - cantidad_spam
print('Correos NO SPAM: ', cantidad_no_spam)

print('=' * 90)

data['text'] = data['text'].str.lower()
data['text'] = data['text'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
data['text'] = data['text'].str.split()
stopwords = list(ENGLISH_STOP_WORDS)
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x if word not in stopwords]))

vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(data['text'])
print('X: ', X.shape)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Datos de entrenamiento: ', X_train.shape)
print('Datos de prueba: ', X_test.shape)

print('=' * 90)

modelo_bayes = MultinomialNB()
modelo_bayes.fit(X_train, y_train)

correos_prueba = pd.read_csv('correos-de-prueba.csv', encoding='latin-1')
print('Registros Externos: ', len(correos_prueba))
correos_prueba['text'] = correos_prueba['text'].str.lower()
correos_prueba['text'] = correos_prueba['text'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
correos_prueba['text'] = correos_prueba['text'].str.split()
correos_prueba['text'] = correos_prueba['text'].apply(lambda x: ' '.join([word for word in x if word not in stopwords]))

X_prueba = vectorizer.transform(correos_prueba['text'])
print('Datos Externos a evaluar: ', X_prueba.shape)

y_pred = modelo_bayes.predict(X_prueba)

y_prueba = correos_prueba['target']

print('=' * 90)

correos_prueba['predicciones'] = np.where(y_pred == 1, "Spam", "No Spam")
print(correos_prueba[['text', 'predicciones']])

precision = accuracy_score(y_prueba, y_pred)
reporte_clasificacion = classification_report(y_prueba, y_pred)

print('=' * 90)

print(f'Precisión del modelo en datos de prueba: {precision:.4f} = {precision * 100:.2f}%')

print('=' * 90)

print('Reporte de clasificación en datos de prueba:')
print(reporte_clasificacion)

print('=' * 90)

precision = accuracy_score(y_prueba, y_pred)
recuperacion = recall_score(y_prueba, y_pred)
print('Precisión (Naive Bayes): ', precision, ' = ', round(precision * 100, 2), '%')
print('Recuperación (Naive Bayes): ', recuperacion, ' = ', round(recuperacion * 100, 2), '%')

