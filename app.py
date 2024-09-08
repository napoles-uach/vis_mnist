import streamlit as st
import pandas as pd
import plotly.express as px

# URL del archivo CSV en GitHub (asegúrate de reemplazar esta URL con la tuya)
url = "https://raw.githubusercontent.com/tu_usuario/tu_repositorio/main/mnist_tsne_results.csv"

@st.cache_data
def load_tsne_data():
    # Leer el archivo CSV desde GitHub
    tsne_data = pd.read_csv(url)
    return tsne_data

# Cargar los datos t-SNE desde el archivo CSV
tsne_data = load_tsne_data()

# Crear la gráfica interactiva con Plotly
fig = px.scatter(tsne_data, x='X', y='Y', color=tsne_data['label'].astype(str),
                 labels={'color': 'Dígito'}, title="t-SNE Visualización del conjunto MNIST")

# Mostrar la gráfica en Streamlit
st.plotly_chart(fig)
