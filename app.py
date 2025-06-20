import streamlit as st
from textblob import TextBlob
import pandas as pd
from datetime import datetime
from deep_translator import GoogleTranslator
from pymongo import MongoClient
import json
from bson import json_util
import nltk
from PIL import Image

# Constantes
CATEGORIAS = ["Electrodomésticos", "Ropa", "Jugueteria", "Hogar", "Calzado","Tecnología"]
CANALES = ["Web", "Redes Sociales", "Call Center", "Tienda física"]
UMBRAL_POSITIVO = 0.08
UMBRAL_NEGATIVO = -0.08

# Configuración de MongoDB
MONGO_URI = "mongodb+srv://mariamartinezb14:Yiwanoz95@cluster0.ojvcfu8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "analisis_sentimientos"
COLLECTION_NAME = "historial_comentarios"

# --------------------------
# Funciones de MongoDB
# --------------------------
def conectar_mongodb():
    """Establece conexión con MongoDB"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        return db[COLLECTION_NAME]
    except Exception as e:
        st.error(f"Error al conectar con MongoDB: {str(e)}")
        return None

def guardar_en_mongodb(registro):
    """Guarda un registro en MongoDB"""
    collection = conectar_mongodb()
    if collection is not None:
        try:
            return collection.insert_one(registro).inserted_id
        except Exception as e:
            st.error(f"Error al guardar en MongoDB: {str(e)}")
    return None

def cargar_historial_desde_mongodb():
    """Carga el historial desde MongoDB"""
    collection = conectar_mongodb()
    if collection is not None:
        try:
            return list(collection.find().sort("Fecha", -1))
        except Exception as e:
            st.error(f"Error al cargar desde MongoDB: {str(e)}")
    return []

# --------------------------
# Funciones de análisis
# --------------------------
def traducir_a_ingles(texto):
    """Traduce texto al inglés usando deep_translator"""
    try:
        if not isinstance(texto, str) or not texto.strip():
            return ""
        
        if all(ord(c) < 128 for c in texto):
            return texto
            
        return GoogleTranslator(source='auto', target='en').translate(texto) or texto
    except Exception as e:
        st.warning(f"Error en traducción: {str(e)}")
        return texto

def analizar_sentimiento(texto):
    """Analiza el sentimiento del texto"""
    texto_traducido = traducir_a_ingles(texto)
    polaridad = TextBlob(texto_traducido).sentiment.polarity
    
    if polaridad > UMBRAL_POSITIVO:
        return "Positivo", polaridad, texto_traducido
    elif polaridad < UMBRAL_NEGATIVO:
        return "Negativo", polaridad, texto_traducido
    return "Neutro", polaridad, texto_traducido

# --------------------------
# Funciones de historial
# --------------------------
def inicializar_historial():
    """Inicializa el historial en session_state"""
    if 'historial' not in st.session_state:
        st.session_state.historial = []
        
        # Cargar datos existentes de MongoDB
        for item in cargar_historial_desde_mongodb():
            agregar_al_historial(
                item['Comentario_Original'],
                item['Categoria'],
                item['Canal'],
                item['Sentimiento'],
                item['Polaridad'],
                item.get('Id_Cliente', 'N/A'),  # Usar 'N/A' si no existe
                item['Fecha']
            )

def agregar_al_historial(comentario, categoria, canal, sentimiento, polaridad, id_cliente='N/A', fecha=None):
    """Agrega un registro al historial"""
    fecha = fecha or datetime.now()
    registro = {
        "Id_Cliente": id_cliente,
        "Fecha": fecha.strftime("%Y-%m-%d %H:%M"),
        "Comentario_Original": comentario[:100] + "..." if len(comentario) > 100 else comentario,
        "Categoría": categoria,
        "Canal": canal,
        "Sentimiento": sentimiento,
        "Polaridad": f"{polaridad:.2f}" if isinstance(polaridad, (float, int)) else polaridad
    }
    st.session_state.historial.insert(0, registro)
    return registro

def mostrar_historial():
    """Muestra el historial en un DataFrame ordenado"""
    if st.session_state.historial:
        df = pd.DataFrame(st.session_state.historial)
        df['Fecha_dt'] = pd.to_datetime(df['Fecha'])
        df = df.sort_values("Fecha_dt", ascending=False).drop('Fecha_dt', axis=1)
        st.dataframe(df, hide_index=True, use_container_width=True)

# --------------------------
# Interfaz de usuario
# --------------------------
st.set_page_config(page_title="🔎Analizador de comentarios", layout="centered")
st.title("🔎Analizador de comentarios - Grupo EXITO")

st.markdown("""
<style>
    /* Color de fondo amarillo */
    .stApp {
        background-color: #FFFF00;
    }
    
    /* Contenedores principales en negro */
    .stContainer, .css-1lcbmhc, .stTextArea, .stSelectbox, .stDataFrame {
        background-color: #000000 !important;
        color: white !important;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Texto en blanco para contraste */
    .stTextInput, .stTextArea, .stSelectbox, .stMarkdown, .stDataFrame {
        color: white !important;
    }
    
    /* Títulos y encabezados */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* Botones personalizados */
    .stButton>button {
        background-color: #000000;
        color: white;
        border: 2px solid white;
        border-radius: 5px;
    }
    
    /* Expanders personalizados */
    .stExpander {
        background-color: #000000;
        border: 1px solid white;
    }
    
    /* Métricas personalizadas */
    .stMetric {
        background-color: #000000;
        color: white;
        border: 1px solid white;
        border-radius: 5px;
    }
    
    /* Cambiar color del texto en los selectboxes */
    .stSelectbox label {
        color: white !important;
    }
    
    /* Cambiar color del texto en los textareas */
    .stTextArea label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Formulario principal
with st.form("formulario_analisis"):
    
    id_cliente = st.text_input("ID del Cliente (opcional):", placeholder="Ingrese el ID del cliente si está disponible")
    
    comentario = st.text_area("Ingrese el comentario:", height=150, 
                            placeholder="Escribe aquí el comentario del cliente...")
    
    col1, col2 = st.columns(2)
    with col1:
        categoria = st.selectbox("Categoría del producto:", CATEGORIAS)
    with col2:
        canal = st.selectbox("Canal de contacto:", CANALES)
    
    
    
    if st.form_submit_button("Analizar Comentario"):
        if comentario.strip():
            with st.spinner("Analizando comentario..."):
                sentimiento, polaridad, texto_traducido = analizar_sentimiento(comentario)
            
            # Mostrar resultados
            color = "green" if sentimiento == "Positivo" else "red" if sentimiento == "Negativo" else "gray"
            st.markdown(f"### Resultado: <span style='color:{color}'>{sentimiento}</span> (Polaridad: {polaridad:.2f})", 
                       unsafe_allow_html=True)
            
            with st.expander("🔍 Ver detalles de traducción"):
                st.write("**Texto original:**", comentario)
                st.write("**Texto traducido (inglés):**", texto_traducido)
            
            # Guardar en MongoDB y actualizar historial
            registro_mongo = {
                "Id_Cliente": id_cliente if id_cliente else 'N/A',
                "Fecha": datetime.now(),
                "Comentario_Original": comentario,
                "Texto_Traducido": texto_traducido,
                "Categoria": categoria,
                "Canal": canal,
                "Sentimiento": sentimiento,
                "Polaridad": polaridad,
                "Metadata": {
                    "longitud_texto": len(comentario),
                    "es_ingles": all(ord(c) < 128 for c in comentario)
                }
            }
            guardar_en_mongodb(registro_mongo)
            
            inicializar_historial()
            agregar_al_historial(comentario, categoria, canal, sentimiento, polaridad, id_cliente if id_cliente else 'N/A')
            
            st.subheader("📋 Historial de Análisis")
            mostrar_historial()
        else:
            st.warning("Por favor ingrese un comentario válido")

# Estadísticas
if 'historial' in st.session_state and st.session_state.historial:
    st.subheader("📊 Estadísticas Globales")
    df_stats = pd.DataFrame(st.session_state.historial)
    
    cols = st.columns(3)
    cols[0].metric("Total análisis", len(df_stats))
    cols[1].metric("Positivos", len(df_stats[df_stats['Sentimiento'] == 'Positivo']))
    cols[2].metric("Negativos", len(df_stats[df_stats['Sentimiento'] == 'Negativo']))

# # Exportar a JSON
# if st.button("📤 Exportar historial completo a JSON"):
#     collection = conectar_mongodb()
#     if collection is not None:
#         try:
#             historial_completo = list(collection.find({}, {"_id": 0}).sort("Fecha", -1))
#             st.download_button(
#                 label="Descargar JSON",
#                 data=json.dumps(historial_completo, indent=2, default=json_util.default),
#                 file_name="historial_comentarios_completo.json",
#                 mime="application/json"
#             )
#         except Exception as e:
#             st.error(f"Error al exportar datos: {str(e)}")
    

