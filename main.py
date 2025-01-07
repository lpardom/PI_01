import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Usa el puerto especificado por Render o 8000 por defecto
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

from fastapi import FastAPI
from datetime import datetime
import pandas as pd

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
def score_titulo(titulo: str):
    film = movies_data[movies_data["original_title"].str.lower() == titulo.lower()]

    if film.empty:
        return {"error": "Película no encontrada"}

    film = film.iloc[0]
    return {
        "titulo": film["original_title"],
        "año": film["release_year"],
        "score": film["popularity"]
    }

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    film = movies_data[movies_data["original_title"].str.lower() == titulo.lower()]

    if film.empty:
        return {"error": "Película no encontrada"}

    film = film.iloc[0]
    vote_count = film["vote_count"]

    if vote_count < 2000:
        return {"message": "La película no cumple con el mínimo de 2000 valoraciones"}

    return {
        "titulo": film["original_title"],
        "año": film["release_year"],
        "total_votos": vote_count,
        "promedio_votos": film["vote_average"]
    }

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
def get_director(nombre_director: str):
    director_data = movies_data[movies_data["director"].str.contains(nombre_director, case=False, na=False)]

    if director_data.empty:
        return {"error": "Director no encontrado"}

    films = []
    for _, row in director_data.iterrows():
        films.append({
            "titulo": row["original_title"],
            "fecha_lanzamiento": row["release_date"].strftime('%Y-%m-%d'),
            "retorno_individual": row["return"],
            "costo": row["budget"],
            "ganancia": row["revenue"] - row["budget"]
        })

    return {
        "director": nombre_director,
        "cantidad_peliculas": director_data.shape[0],
        "peliculas": films
    }
