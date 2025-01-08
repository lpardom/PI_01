# Proyecto Individual: Sistema de Recomendación de Películas

Este proyecto tiene como objetivo desarrollar un **MVP (Producto Mínimo Viable)** de un sistema de recomendación de películas utilizando datos disponibles en los archivos `movies_dataset.csv` y `credits.csv`. Como MLOps Engineer, el objetivo es llevar un modelo de Machine Learning al entorno real, integrando procesos de limpieza de datos, desarrollo de API y despliegue.

## Contexto del Proyecto
La startup para la que trabajas ofrece servicios de agregación de plataformas de streaming. Actualmente, los datos no están estructurados y no existen procesos automatizados. El objetivo es realizar las siguientes tareas:

1. **ETL y limpieza de datos**.
2. **Desarrollo de una API** para disponibilizar los datos procesados.
3. **Entrenamiento y despliegue de un sistema de recomendación** basado en similitud.
4. **Visualización del trabajo en un video corto.**

---

## Pasos del Desarrollo

### 1. **Limpieza y Transformación de Datos**
Las siguientes transformaciones se realizaron para preparar los datos:

- Desanidar campos como belongs_to_collection, genres, production_companies, production_countries y spoken_languages para permitir su uso en consultas.
- Rellenar valores nulos:
  - `revenue` y `budget` con `0`.
  - Eliminar filas con valores nulos en `release_date`.
- Formatear las fechas al formato `AAAA-mm-dd` y crear una columna `release_year` basada en `release_date`.
- Crear la columna `return` (retorno de inversión) calculada como `revenue / budget`, asignando `0` cuando no sea posible calcularlo.
- Eliminar columnas innecesarias: `video`, `imdb_id`, `adult`, `original_title`, `poster_path`, `homepage`.

### 2. **Desarrollo de la API con FastAPI**
Se creó una API con las siguientes funcionalidades:

#### Endpoints Disponibles

1. **`/cantidad_filmaciones_mes/{mes}`**: Devuelve la cantidad de películas estrenadas en un mes ingresado (en español).
2. **`/cantidad_filmaciones_dia/{dia}`**: Devuelve la cantidad de películas estrenadas en un día ingresado (en español).
3. **`/score_titulo/{titulo}`**: Devuelve el título, año de estreno y popularidad de una película.
4. **`/votos_titulo/{titulo}`**: Devuelve el título, número de votos y promedio de votos de una película, considerando un mínimo de 2000 votos.
5. **`/get_actor/{nombre_actor}`**: Devuelve el éxito de un actor, incluyendo cantidad de películas, retorno total y promedio por filmación.
6. **`/get_director/{nombre_director}`**: Devuelve el éxito de un director, incluyendo detalles como retorno total, películas dirigidas y ganancia por filmación.
7. **`/recomendacion/{titulo}`**: Devuelve una lista de 5 películas similares a la ingresada, basada en un modelo de similitud.

### 3. **Entrenamiento del Sistema de Recomendación**
El sistema de recomendación utiliza la similitud de coseno para encontrar películas similares. El preprocesamiento combina información relevante de géneros, sinopsis y frases promocionales para crear vectores de características, los cuales son comparados entre sí.

#### Flujo del Modelo
1. Crear una columna `combined_features` combinando:
   - `genre_names`, `overview`, y `tagline`.
2. Vectorizar los textos usando `CountVectorizer`.
3. Calcular la matriz de similitud de coseno.
4. Retornar las 5 películas con mayor puntuación de similitud.

### 4. **Despliegue**
El modelo y la API se desplegaron en la nube utilizandon **FastAPI** y **Render** para que puedan ser consumidos de manera pública.

### 5. **Análisis Exploratorio de Datos (EDA)**
Se realizaron gráficas para explorar los datos:
- Distribuciones de popularidad y cantidad de votos.
- Nube de palabras para analizar las palabras más frecuentes en los títulos.

---

## Estructura del Repositorio

- 01 ETL : Es el notebook con todo el proceso de ETL.
- 02 EDA : Es el notebook con el proceso de EDA.
- `main.py`: Contiene la implementación de la API con FastAPI.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.
- `movies.csv` que es el dataset resultante del proceso de ETL y combinacion de `movies_dataset.csv` y `credits.csv`: Datos utilizados en el proyecto.
- `README.md`: Documentación del proyecto (este archivo).

---

## Tecnologías Utilizadas

- **Python**: Lenguaje principal para el desarrollo.
- **Pandas**: Limpieza y manipulación de datos.
- **Scikit-learn**: Vectorización y cálculo de similitud.
- **FastAPI**: Framework para la creación de APIs.
- **Render**: Servicio de despliegue en la nube.

---

## Video Demostrativo
Un video de menos de 7 minutos mostrando:
1. El funcionamiento de los endpoints de la API.
2. Un breve resumen del sistema de recomendación.
3. Análisis Exploratorio y desarrollo del modelo.

Link del video: [Insertar enlace aquí]

---

## Enlaces para consulta

1. Github : [https://github.com/lpardom/PI_01]
  
2. Render: [https://pi-01-g2xb.onrender.com/docs]
  


---
