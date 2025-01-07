from fastapi import FastAPI
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Cargar el dataset
movies_data = pd.read_csv("movies.csv")

# Convertir las fechas de lanzamiento a un formato datetime
movies_data["release_date"] = pd.to_datetime(movies_data["release_date"], errors="coerce")

# Mapeos para los nombres de meses y días en español
days_es = {"lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, "viernes": 4, "sábado": 5, "domingo": 6}
months_es = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
}

@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    mes = mes.lower()
    if mes not in months_es:
        return {"error": "Mes no válido"}

    month_number = months_es[mes]
    count = movies_data[movies_data["release_date"].dt.month == month_number].shape[0]

    return {"message": f"{count} cantidad de películas fueron estrenadas en el mes de {mes}"}

@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dia = dia.lower()
    if dia not in days_es:
        return {"error": "Día no válido"}

    day_number = days_es[dia]
    count = movies_data[movies_data["release_date"].dt.dayofweek == day_number].shape[0]

    return {"message": f"{count} cantidad de películas fueron estrenadas en los días {dia}"}

@app.get("/score_titulo/{titulo}")
def score_titulo(titulo_de_la_filmacion):
    # Filtrar la película por el título, ignorando mayúsculas y minúsculas
    filtro = movies_data['title'].str.lower() == titulo_de_la_filmacion.lower()
    pelicula = movies_data.loc[filtro]
    
    # Verificar si se encontró la película
    if pelicula.empty:
        return f"La película '{titulo_de_la_filmacion}' no fue encontrada."
    
    # Obtener los datos de la película
    pelicula = pelicula.iloc[0]  # Seleccionar la primera coincidencia
    titulo = pelicula['title']
    anio = pelicula['release_year']
    score = pelicula['popularity']
    
    # Formatear el mensaje de retorno
    return f"La película '{titulo}' fue estrenada en el año {anio} con un score/popularidad de {score:.2f}"

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo_de_la_filmacion):
    # Filtrar la película por el título, ignorando mayúsculas y minúsculas
    filtro = movies_data['title'].str.lower() == titulo_de_la_filmacion.lower()
    pelicula = movies_data.loc[filtro]
    
    # Verificar si se encontró la película
    if pelicula.empty:
        return f"La película '{titulo_de_la_filmacion}' no fue encontrada."
    
    # Obtener los datos de la película
    pelicula = pelicula.iloc[0]  # Seleccionar la primera coincidencia
    titulo = pelicula['title']
    anio = pelicula['release_year']
    votos = pelicula['vote_count']
    promedio_votos = pelicula['vote_average']
    
    # Verificar si cumple con el mínimo de votos
    if votos < 2000:
        return f"La película '{titulo}' no cumple con el mínimo de 2000 valoraciones (tiene {votos} votos)."
    
    # Formatear el mensaje de retorno
    return (f"La película '{titulo}' fue estrenada en el año {anio}. "
            f"La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio_votos:.2f}.")

@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    actor_data = movies_data[movies_data["actor_names"].str.contains(nombre_actor, case=False, na=False)]

    if actor_data.empty:
        return {"error": "Actor no encontrado"}

    total_ret = actor_data["return"].sum()
    count_films = actor_data.shape[0]
    avg_ret = total_ret / count_films

    return {
        "actor": nombre_actor,
        "cantidad_filmaciones": count_films,
        "retorno_total": total_ret,
        "retorno_promedio": avg_ret
    }

@app.get("/get_director/{nombre_director}")
def get_director(nombre_director):
    # Filtrar las películas dirigidas por el director (ignorando mayúsculas y minúsculas)
    filtro = movies_data['director'].str.contains(nombre_director, case=False, na=False)
    peliculas = movies_data.loc[filtro]
    
    # Verificar si se encontró el director
    if peliculas.empty:
        return f"El director '{nombre_director}' no fue encontrado o no tiene películas en el dataset."
    
    # Calcular el retorno total y organizar los detalles de las películas
    total_retorno = peliculas['return'].sum()
    detalles_peliculas = []
    
    for _, pelicula in peliculas.iterrows():
        titulo = pelicula['title']
        fecha_lanzamiento = pelicula['release_date']
        retorno_individual = pelicula['return']
        costo = pelicula['budget']
        ganancia = pelicula['revenue'] - costo if not pd.isna(pelicula['revenue']) else -costo  # Ganancia puede ser negativa si no hay ingresos

        detalles_peliculas.append({
            "titulo": titulo,
            "fecha_lanzamiento": fecha_lanzamiento,
            "retorno_individual": round(retorno_individual, 2) if not pd.isna(retorno_individual) else None,
            "costo": costo,
            "ganancia": ganancia
        })
    
    # Formatear el mensaje de retorno
    return {
        "director": nombre_director,
        "retorno_total": round(total_retorno, 2),
        "peliculas": detalles_peliculas
    }

    # Preprocesamiento para el sistema de recomendación
def preprocess_data(df):
    df['combined_features'] = df['title'] + " " + df['genre_names'] + " " + df['overview']
    return df

# Crear matriz de similitud
def create_similarity_matrix(df):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform(df['combined_features'])
    similarity_matrix = cosine_similarity(count_matrix, count_matrix)
    return similarity_matrix

# Función de recomendación
def recomendacion(titulo, df, similarity_matrix):
    if titulo not in df['title'].values:
        return ["Película no encontrada"]

    movie_index = df[df['title'] == titulo].index[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [movie[0] for movie in sorted_movies[1:6]]
    recommended_titles = df['title'].iloc[recommended_indices].tolist()
    return recommended_titles