from fastapi import FastAPI
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Cargar el dataset
# movies_data = pd.read_csv("movies.csv")

# Preprocesamiento: combinar información relevante en una sola columna para similitud
def preprocess_data(df):
    df['combined_features'] = (
        df['genre_names'].fillna('') + " " +
        df['overview'].fillna('') + " " +
        df['tagline'].fillna('')
    )
    return df

movies_data = pd.DataFrame({
    "title": ["Toy Story", "Jumanji", "Grumpier Old Men"],
    "genre_names": ["Animation, Comedy, Family", "Adventure, Fantasy, Family", "Comedy, Romance"],
    "overview": [
        "A story about toys coming to life.",
        "A magical board game unleashes adventure.",
        "A continuation of a comedic love story."
    ],
    "tagline": [
        "The adventure takes off!",
        "Roll the dice and unleash the excitement!",
        "Still yummy after all these years."
    ],
    "release_date": pd.to_datetime(["1995-11-22", "1995-12-15", "1995-12-22"]),
    "release_year": [1995, 1995, 1995],
    "popularity": [21.95, 17.02, 3.85],
    "vote_count": [5415, 2413, 34],
    "vote_average": [8.0, 7.5, 6.2],
    "actor_names": [
        "Tom Hanks, Tim Allen, Don Rickles",
        "Robin Williams, Kirsten Dunst, Jonathan Hyde",
        "Jack Lemmon, Walter Matthau, Ann-Margret"
    ],
    "director": ["John Lasseter", "Joe Johnston", "Howard Deutch"],
    "budget": [30000000.0, 65000000.0, 16000000.0],
    "revenue": [373554033.0, 262797249.0, 81452156.0],
    "return": [12.45, 4.04, 5.09],
    "combined_features": [
        "Animation Comedy Family A story about toys coming to life. The adventure takes off!",
        "Adventure Fantasy Family A magical board game unleashes adventure. Roll the dice and unleash the excitement!",
        "Comedy Romance A continuation of a comedic love story. Still yummy after all these years."
    ]
})

movies_df = preprocess_data(movies_data)

# Construcción de la matriz de características de texto
count_vectorizer = CountVectorizer(stop_words='english')
feature_matrix = count_vectorizer.fit_transform(movies_df['combined_features'])

# Calcular la similitud de coseno
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# Función de recomendación
def recomendacion(titulo):
    # Convertir el título ingresado a minúsculas
    titulo = titulo.lower()

    # Crear una columna temporal con títulos en minúsculas para comparación
    movies_data['title_lower'] = movies_data['title'].str.lower()

    # Verificar si el título existe en la base de datos
    if titulo not in movies_data['title_lower'].values:
        return ["La película no fue encontrada en la base de datos"]

    # Encontrar el índice de la película según el título en minúsculas
    idx = movies_data[movies_data['title_lower'] == titulo].index[0]

    # Obtener las puntuaciones de similitud de la película con todas las demás
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar por puntuación de similitud, excluyendo la misma película
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similares

    # Obtener los títulos de las películas recomendadas
    recommended_movies = [movies_df.iloc[i[0]]['title'] for i in sim_scores]
    
    # Eliminar la columna temporal para no afectar los datos originales
    movies_data.drop(columns=['title_lower'], inplace=True)
    
    return recommended_movies


app = FastAPI()
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
    columns_to_load = ['release_date']
    movies_data = pd.read_csv("movies.csv", usecols=columns_to_load)     

    mes = mes.lower()
    if mes not in months_es:
        return {"error": "Mes no válido"}

    month_number = months_es[mes]
    count = movies_data[movies_data["release_date"].dt.month == month_number].shape[0]

    return {"message": f"{count} cantidad de películas fueron estrenadas en el mes de {mes}"}

@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    columns_to_load = ['release_date']
    movies_data = pd.read_csv("movies.csv", usecols=columns_to_load)
    dia = dia.lower()
    if dia not in days_es:
        return {"error": "Día no válido"}

    day_number = days_es[dia]
    count = movies_data[movies_data["release_date"].dt.dayofweek == day_number].shape[0]

    return {"message": f"{count} cantidad de películas fueron estrenadas en los días {dia}"}

@app.get("/score_titulo/{titulo}")
def score_titulo(titulo_de_la_filmacion):
    columns_to_load = ['title','release_year','popularity']
    movies_data = pd.read_csv("movies.csv", usecols=columns_to_load)
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
    columns_to_load = ['title','release_year','popularity','vote_count','vote_average']
    movies_data = pd.read_csv("movies.csv", usecols=columns_to_load)

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
    columns_to_load = ['actor_names','return','title']
    movies_data = pd.read_csv("movies.csv", usecols=columns_to_load)

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
    columns_to_load = ['director','title','release_date','return','budget','revenue']
    movies_data = pd.read_csv("movies.csv", usecols=columns_to_load)

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

@app.get("/recomendacion/{titulo}")
def get_recomendacion(titulo: str):
        columns_to_load = ['title', 'genre_names', 'overview', 'tagline']
        movies_data = pd.read_csv("movies.csv", usecols=columns_to_load)
        # movies_data = pd.read_csv("movies.csv")
        return {"recomendaciones": recomendacion(titulo)}