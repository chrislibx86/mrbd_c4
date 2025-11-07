import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

# --------------------------------------------------------------------------------------
# 1. Conexión con MongoDB
# --------------------------------------------------------------------------------------
DB_NAME = "p3_marcosref_reclibros"

def conexion_mongo():
    uri = "mongodb+srv://temporal_1:MPMXmpxTQnDB2ph1@cluster0.icuj9td.mongodb.net/?appName=Cluster0"
    cliente = MongoClient(uri, server_api=ServerApi('1'))
    return cliente[DB_NAME]

db = conexion_mongo()

# --------------------------------------------------------------------------------------
# 2. Cargar libros y usuarios desde MongoDB
# --------------------------------------------------------------------------------------
df_libros = pd.DataFrame(list(db["libros"].find({}, {"_id": 0})))


df_libros.columns = df_libros.columns.str.strip().str.lower()


columnas_requeridas = {"nombre", "complejidad", "popularidad"}
if not columnas_requeridas.issubset(set(df_libros.columns)):
    st.error(f"❌ Las columnas necesarias no están presentes en la colección 'libros'.\nColumnas encontradas: {list(df_libros.columns)}")
    st.stop()


if "usuarios" not in db.list_collection_names():
    usuarios_default = [
        {"nombre": n} for n in ["Freddy", "Eduardo", "Jimmy"]
    ]
    db["usuarios"].insert_many(usuarios_default)


usuarios = [u["nombre"] for u in db["usuarios"].find({}, {"_id": 0, "nombre": 1})]
nombres_libros = df_libros["nombre"].tolist()

# --------------------------------------------------------------------------------------
# 3. Función de cálculo de similitud y guardado en MongoDB
# --------------------------------------------------------------------------------------
def calcular_similitud(df_calificaciones, df_relevancia, df_profundidad, df_caracteristicas_libro):
    calificaciones_transp = df_calificaciones.T
    relevancia_transp = df_relevancia.T
    profundidad_transp = df_profundidad.T

    calificaciones_transp.columns = [f"{u}_calif" for u in usuarios]
    relevancia_transp.columns = [f"{u}_relev" for u in usuarios]
    profundidad_transp.columns = [f"{u}_prof" for u in usuarios]

    temp_dinamica = pd.concat([calificaciones_transp, relevancia_transp, profundidad_transp], axis=1)
    libros_comunes = list(set(temp_dinamica.index) & set(df_caracteristicas_libro.index))

    dinamicas = temp_dinamica.loc[libros_comunes]
    estaticas = df_caracteristicas_libro.loc[libros_comunes]

    combinadas = pd.concat([dinamicas, estaticas], axis=1)
    matriz_similitud = pd.DataFrame(
        cosine_similarity(combinadas),
        index=combinadas.index,
        columns=combinadas.index
    )

    db["similitud_libros"].delete_many({})
    registros = []
    for libro in matriz_similitud.index:
        similares = matriz_similitud.loc[libro].sort_values(ascending=False)[1:6]
        registros.append({
            "base": libro,
            "similares": [{"nombre": n, "similitud": float(sim)} for n, sim in similares.items()]
        })
    if registros:
        db["similitud_libros"].insert_many(registros)

    return matriz_similitud

# --------------------------------------------------------------------------------------
# 4. Configuración inicial del modelo
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Eduardo Mendieta - Recomendador de libros MongoDB", layout="wide")

if "calificaciones" not in st.session_state:
    shape = (len(usuarios), len(nombres_libros))
    st.session_state.calificaciones = pd.DataFrame(np.random.randint(1, 6, shape), index=usuarios, columns=nombres_libros)
    st.session_state.relevancia = pd.DataFrame(np.random.randint(1, 6, shape), index=usuarios, columns=nombres_libros)
    st.session_state.profundidad = pd.DataFrame(np.random.randint(1, 6, shape), index=usuarios, columns=nombres_libros)

    df_caracteristicas = (
        df_libros[["nombre", "complejidad", "popularidad"]]
        .rename(columns={
            "nombre": "Libro",
            "complejidad": "Complejidad de Lectura",
            "popularidad": "Popularidad Global"
        })
        .set_index("Libro")
    )

    st.session_state.caracteristicas_libro = df_caracteristicas
    st.session_state.similitud = calcular_similitud(
        st.session_state.calificaciones,
        st.session_state.relevancia,
        st.session_state.profundidad,
        st.session_state.caracteristicas_libro
    )

# --------------------------------------------------------------------------------------
# 5. Capa Batch
# --------------------------------------------------------------------------------------
st.title("📚 Recomendador de libros con MongoDB y arquitectura Lambda")
st.caption("Capa Batch + Velocidad + Servicio con persistencia de datos")

st.subheader("🧩 Capa Batch - Datos base")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Calificaciones históricas simuladas**")
    st.dataframe(st.session_state.calificaciones)

with col2:
    st.markdown("**Características estáticas de libros (desde MongoDB)**")
    st.dataframe(st.session_state.caracteristicas_libro)

# --------------------------------------------------------------------------------------
# 6. Capa de velocidad
# --------------------------------------------------------------------------------------
st.header("⚡ Capa de velocidad - Calificación en tiempo real")

usuario = st.selectbox("Selecciona un usuario:", usuarios)
libro = st.selectbox("Selecciona un libro:", nombres_libros)

calif = st.slider("Calificación (1-5)", 1, 5, 3)
relev = st.slider("Relevancia personal (1-5)", 1, 5, 3)
prof = st.slider("Profundidad de lectura (1-5)", 1, 5, 3)

if st.button("➕ Guardar opinión y recalcular modelo"):
    st.session_state.calificaciones.loc[usuario, libro] = calif
    st.session_state.relevancia.loc[usuario, libro] = relev
    st.session_state.profundidad.loc[usuario, libro] = prof

    db["opiniones"].insert_one({
        "usuario": usuario,
        "libro": libro,
        "calificacion": calif,
        "relevancia": relev,
        "profundidad": prof,
        "fecha": datetime.utcnow().isoformat()
    })

    st.session_state.similitud = calcular_similitud(
        st.session_state.calificaciones,
        st.session_state.relevancia,
        st.session_state.profundidad,
        st.session_state.caracteristicas_libro
    )

    st.success(f"✅ Opinión de {usuario} sobre '{libro}' guardada y similitud actualizada.")

# --------------------------------------------------------------------------------------
# 7. Capa de servicio
# --------------------------------------------------------------------------------------
st.header("🛰️ Capa de servicio - Recomendaciones desde MongoDB")

libro_sel = st.selectbox("Selecciona un libro base:", nombres_libros)
doc = db["similitud_libros"].find_one({"base": libro_sel}, {"_id": 0})

if doc:
    st.write(f"🔍 Libros más similares a **{libro_sel}**:")
    st.dataframe(pd.DataFrame(doc["similares"]))
else:
    st.warning("⚠️ No se encontró similitud almacenada para este libro.")

st.info("**Estudiante:** Eduardo Mendieta.")
