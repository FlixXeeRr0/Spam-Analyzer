# TAREA 4 - MODELO DE APRENDIZAJE AUTOMATICO
**INSTITUTO TECNOLOGICO DE CULIACAN**

**ASIGNATURA:**
* *Inteligencia Artificial*

**DOCENTE:**
* *Mora Felix Zuriel Dathan*

**INTEGRANTES DEL EQUIPO:**
* *Cardoza Perez Agustin*
* *Ramirez Moreno Ivan Eduardo*

**HORARIO:**
* *16:00 - 17:00*

**ACTIVIDAD:**
* *Clasificación de Correos Electrónicos: Spam vs No Spam*

## Objetivo:
Desarrollar un modelo de aprendizaje automático para clasificar correos electrónicos como “Spam” o “No Spam” utilizando el teorema de Bayes.

## Requisitos:
- Python 3.x
- **Librerías:**
    - Numpy.
    - Pandas.
    - Sklearn.

## Datos:
Conjunto de datos [SpamAssassin Public Corpus](https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset) que contiene 5827 correos etiquetados como “Spam = 1” o “No spam = 0”.

## Preprocesamiento de Datos:
1. Eliminar correos electrónicos duplicados.
2. Convertir todo el texto a minúsculas.
3. Eliminar caracteres especiales.
4. Eliminar palabras vacías (stopwords).
5. Tokenizar el texto.

## Desarrollo:

### 1. Extracción de Características:
- **Frecuencia de términos (TF):** 
    * *Contar el número de veces que aparece cada palabra en un correo electrónico.*
- **Frecuencia de términos invertida (IDF):** 
    * *Medir la importancia de cada palabra en el conjunto de datos.*
- **TF-IDF:** 
    * *Combinar TF e IDF para obtener una medida ponderada de la importancia de cada palabra.*

### 2. Modelo:
El modelo se basa en el teorema de Bayes para calcular la probabilidad de que un correo electrónico sea spam dado su conjunto de características.

### 3. Algoritmo (Código en Python):

- **Calcular la probabilidad previa de spam (P(Spam)):**
    ```python
    P_spam = numero_correos_spam / total_correos
    ```

- **Calcular la probabilidad de las características del correo electrónico dado que es spam (P(Características|Spam)):**
    ```python
    P_caracteristicas_spam = frecuencia_caracteristicas_spam / total_caracteristicas_spam
    ```

- **Calcular la probabilidad de las características del correo electrónico dado que no es spam (P(Características|NoSpam)):**
    ```python
    P_caracteristicas_no_spam = frecuencia_caracteristicas_no_spam / total_caracteristicas_no_spam
    ```

- **Calcular la probabilidad posterior de que el correo electrónico sea spam (P(Spam|Características)):**
    ```python
    P_spam_caracteristicas = (P_spam * P_caracteristicas_spam) / 
    (P_spam * P_caracteristicas_spam + P_no_spam * P_caracteristicas_no_spam)
    ```

- **Clasificación:**
    Un correo electrónico se clasificará como spam si:
    ```python
    P(Spam|Características) > P(NoSpam|Características)
    ```

- **Evaluación:**
    - **Precisión:**
      ```python
      precision = np.sum(clasificaciones == data[“spam”])/len(clasificaciones)
      ```
    - **Recuperación:**
      ```python
      recuperacion = np.sum(clasificaciones == data[“spam”])/data[“spam”].sum()
      ```
## Modificacion Solicitada:
Se agrega al repositorio un archivo llamado **analizadorModificado.py** el cual tiene cambios en el codigo. Se eliminan los calculos de las probabildades e impresion de tablas.
### Nueva Funcionalidad:
Se agrega funcionalidad extra la cual nos permite analizar un archivo csv externo con nuestros correos y clasificacion para que el modelo entrenado, acorde a su entrenamiento nos prediga si el correo es o no SPAM.
### Archivos Nuevos:
Codigo: ```analizadorModificado.py```
CSV con correos externos: ```correos-de-prueba.csv```