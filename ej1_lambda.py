import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Eduardo Mendieta - Recomendador de libros - Arquitectura Lambda", layout="wide")

# --------------------------------------------------------------------------------------
# 1. Definición de datos base
# --------------------------------------------------------------------------------------

datos_libros = {
    "Cien Años de Soledad": {"categoria": "Ficción Mágica", "complejidad": 5, "popularidad": 4},
    "1984": {"categoria": "Ciencia Ficción", "complejidad": 4, "popularidad": 5},
    "El Principito": {"categoria": "Fábula", "complejidad": 1, "popularidad": 5},
    "Sapiens": {"categoria": "No Ficción", "complejidad": 3, "popularidad": 3},
    "Don Quijote de la Mancha": {"categoria": "Clásico", "complejidad": 5, "popularidad": 3},
    "Dune": {"categoria": "Ciencia Ficción", "complejidad": 4, "popularidad": 4},
    "Orgullo y Prejuicio": {"categoria": "Romance Clásico", "complejidad": 2, "popularidad": 4},
    "El Hobbit": {"categoria": "Fantasía", "complejidad": 3, "popularidad": 5},
    "Crimen y Castigo": {"categoria": "Novela Filosófica", "complejidad": 5, "popularidad": 3},
    "Cumbres Borrascosas": {"categoria": "Romance Gótico", "complejidad": 3, "popularidad": 4},
    "Moby Dick": {"categoria": "Aventura", "complejidad": 4, "popularidad": 2},
    "Matar a un Ruiseñor": {"categoria": "Drama Social", "complejidad": 2, "popularidad": 5},
    "La Carretera": {"categoria": "Post-apocalíptico", "complejidad": 3, "popularidad": 3},
    "Rayuela": {"categoria": "Experimental", "complejidad": 5, "popularidad": 2},
    "El Nombre del Viento": {"categoria": "Fantasía Épica", "complejidad": 4, "popularidad": 4},
    "Una Breve Historia del Tiempo": {"categoria": "Divulgación Científica", "complejidad": 4, "popularidad": 5},
    "Ensayo sobre la Ceguera": {"categoria": "Novela Distópica", "complejidad": 3, "popularidad": 4},
    " Siddhartha": {"categoria": "Filosofía", "complejidad": 2, "popularidad": 3},
    "Drácula": {"categoria": "Terror Clásico", "complejidad": 3, "popularidad": 4},
    "El Padrino": {"categoria": "Crimen", "complejidad": 2, "popularidad": 5},
    "Psicoanalista": {"categoria": "Thriller", "complejidad": 3, "popularidad": 3},
    "Fahrenheit 451": {"categoria": "Ciencia Ficción", "complejidad": 2, "popularidad": 4},
    "La Odisea": {"categoria": "Épica Antigua", "complejidad": 5, "popularidad": 2},
    "El Código Da Vinci": {"categoria": "Misterio", "complejidad": 1, "popularidad": 5},
    "Los Miserables": {"categoria": "Novela Histórica", "complejidad": 5, "popularidad": 3},
    "Las Uvas de la Ira": {"categoria": "Drama Social", "complejidad": 4, "popularidad": 2},
    "Harry Potter 1": {"categoria": "Fantasía Juvenil", "complejidad": 1, "popularidad": 5},
}
nombres_libros = list(datos_libros.keys())
usuarios = ["Ana", "Luis", "Carlos", "María", "Elena", "Javier", "Sofía", "Miguel", "Laura"] # Más usuarios para más varianza
columnas_caracteristicas = ["Complejidad de Lectura", "Popularidad Global"]

def calcular_similitud(df_calificaciones, df_relevancia, df_profundidad, df_caracteristicas_libro):
    calificaciones_transpuestas = df_calificaciones.T
    relevancia_transpuestas = df_relevancia.T
    profundidad_transpuestas = df_profundidad.T

    calificaciones_transpuestas.columns = [f"{u}_calif" for u in usuarios]
    relevancia_transpuestas.columns = [f"{u}_relev" for u in usuarios]
    profundidad_transpuestas.columns = [f"{u}_prof" for u in usuarios]
    
    temp_dinamica = pd.concat([
        calificaciones_transpuestas, 
        relevancia_transpuestas, 
        profundidad_transpuestas
    ], axis=1)

    libros_comunes = list(set(temp_dinamica.index) & set(df_caracteristicas_libro.index))
    
    caracteristicas_dinamicas_alineadas = temp_dinamica.loc[libros_comunes]
    caracteristicas_estaticas_alineadas = df_caracteristicas_libro.loc[libros_comunes]
    
    caracteristicas_combinadas = pd.concat([
        caracteristicas_dinamicas_alineadas, 
        caracteristicas_estaticas_alineadas
    ], axis=1)
    
    matriz_similitud = pd.DataFrame(
        cosine_similarity(caracteristicas_combinadas),
        index=caracteristicas_combinadas.index,
        columns=caracteristicas_combinadas.index
    )
    return matriz_similitud

if 'calificaciones' not in st.session_state:
    initial_shape = (len(usuarios), len(nombres_libros))
    
    st.session_state.calificaciones = pd.DataFrame(
        np.random.randint(1, 6, size=initial_shape),
        index=usuarios,
        columns=nombres_libros
    )
    st.session_state.relevancia = pd.DataFrame(
        np.random.randint(1, 6, size=initial_shape),
        index=usuarios,
        columns=nombres_libros
    )
    st.session_state.profundidad = pd.DataFrame(
        np.random.randint(1, 6, size=initial_shape),
        index=usuarios,
        columns=nombres_libros
    )
    
    lista_caracteristicas = []
    for libro, datos in datos_libros.items():
        lista_caracteristicas.append({
            "Libro": libro,
            "Complejidad de Lectura": datos["complejidad"],
            "Popularidad Global": datos["popularidad"],
        })
    st.session_state.caracteristicas_libro = pd.DataFrame(lista_caracteristicas).set_index("Libro")
    
    st.session_state.similitud = calcular_similitud(
        st.session_state.calificaciones, 
        st.session_state.relevancia,
        st.session_state.profundidad,
        st.session_state.caracteristicas_libro
    )

# --------------------------------------------------------------------------------------
# 2. Capa Batch
# --------------------------------------------------------------------------------------
st.title("📚 Recomendador de libros con arquitectura Lambda")
st.caption("Simulación: Capa Batch + Capa de Velocidad + Capa de Servicio")

st.subheader("🧩 Capa Batch - Datos base")
st.write("La capa batch procesa los datos históricos que cambian lentamente.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Calificaciones Históricas **")
    st.dataframe(st.session_state.calificaciones)

with col2:
    st.markdown("**Características estáticas de libros**")
    st.write("Variables intrínsecas del libro: complejidad y popularidad.")
    st.dataframe(st.session_state.caracteristicas_libro)


st.subheader("🤖 Modelo Batch - Similitud combinada entre libros")
st.write(
    "Calculada mediante Similitud Coseno sobre un vector que incluye: "
    "1. Las 3 métricas dinámicas de todos los usuarios (Calificación, Relevancia, Profundidad)."
    "2. Las 2 métricas estáticas del libro (Complejidad, Popularidad)."
)
st.dataframe(st.session_state.similitud.round(3), use_container_width=True)

# --------------------------------------------------------------------------------------
# 3. Capa de velocidad 
# --------------------------------------------------------------------------------------
st.header("⚡ Capa de velocidad - Ingreso en tiempo real")

usuario_velocidad = st.selectbox("Selecciona un usuario para calificar:", usuarios)
libro_velocidad = st.selectbox("Selecciona un libro para calificar:", nombres_libros)

calificacion_velocidad = st.slider("1. Nueva Calificación (1-5)", 1, 5, 3, key="new_rating")
relevancia_velocidad = st.slider("2. Relevancia Personal (1-5) - ¿Qué tan importante es para ti?", 1, 5, 3, key="new_relevance")
profundidad_velocidad = st.slider("3. Profundidad Cubierta (1-5) - ¿Lo leíste completo o solo por encima?", 1, 5, 3, key="new_depth")


def actualizar_modelo():
    st.session_state.calificaciones.loc[usuario_velocidad, libro_velocidad] = calificacion_velocidad
    st.session_state.relevancia.loc[usuario_velocidad, libro_velocidad] = relevancia_velocidad
    st.session_state.profundidad.loc[usuario_velocidad, libro_velocidad] = profundidad_velocidad
    
    st.session_state.similitud = calcular_similitud(
        st.session_state.calificaciones, 
        st.session_state.relevancia,
        st.session_state.profundidad,
        st.session_state.caracteristicas_libro
    )

if st.button("➕ Agregar opinión"):

    actualizar_modelo()
    st.success(f"✅ ¡Éxito! {usuario_velocidad} ha actualizado las métricas para '{libro_velocidad}'. El modelo se ha actualizado.")

# --------------------------------------------------------------------------------------
# 4. Capa de servicio
# --------------------------------------------------------------------------------------
st.header("🛰️ Capa de Servicio - Recomendaciones actualizadas")

libro_seleccionado = st.selectbox("Selecciona un libro base para encontrar similares:", nombres_libros)

recomendaciones = st.session_state.similitud[libro_seleccionado].sort_values(ascending=False)[1:5]

st.write("🔍 Libros más similares a:", libro_seleccionado)

df_recomendaciones = pd.DataFrame(recomendaciones)
df_recomendaciones.columns = ["Similitud Combinada"]
st.dataframe(df_recomendaciones.round(4), use_container_width=True)

st.info("**Estudiante:** Eduardo Mendieta.")