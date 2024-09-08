import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist
import random
import pandas as pd

# Título de la app en Streamlit
st.title('Visualización de t-SNE con imágenes aleatorias del conjunto MNIST')

# Cargar datos MNIST
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

# Aplanar imágenes y normalizarlas para t-SNE
X_valid_flattened = X_valid.reshape(-1, 28*28) / 255.0

@st.cache_data
def load_tsne_data():
    url = "https://raw.githubusercontent.com/napoles-uach/vis_mnist/main/mnist_tsne_results.csv"
    tsne_data = pd.read_csv(url)
    return tsne_data

# Cargar los datos t-SNE desde GitHub solo una vez
tsne_data = load_tsne_data()

# Crear gráfico interactivo con Plotly
fig = px.scatter(tsne_data, x='X', y='Y', color=y_valid.astype(str),
                 labels={'color': 'Dígito'}, title="t-SNE Visualización del conjunto MNIST")

# Función para convertir imágenes a base64
def encode_image(image):
    fig_mpl = Figure()
    ax = fig_mpl.add_subplot(111)
    ax.imshow(image, cmap="gray")
    ax.axis('off')
    buf = BytesIO()
    fig_mpl.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data

# Seleccionar 10 índices al azar
random_indices = random.sample(range(len(tsne_data)), 10)

# Añadir 10 imágenes al azar sobre los puntos de t-SNE
for index in random_indices:
    position = tsne_data.iloc[index]
    image = X_valid[index]
    image_base64 = encode_image(image)

    # Añadir la imagen sobre el punto t-SNE en la posición calculada
    fig.add_layout_image(
        dict(
            source='data:image/png;base64,{}'.format(image_base64),
            xref="x", yref="y",
            x=position['X'], y=position['Y'],
            sizex=0.10, sizey=0.10,
            xanchor="center", yanchor="middle"
        )
    )

# Configurar el gráfico
fig.update_traces(marker=dict(size=5, opacity=1, line=dict(width=1)))
fig.update_layout(showlegend=False)

# Mostrar la gráfica en Streamlit
st.plotly_chart(fig)


digit_counts = pd.DataFrame(y_train, columns=["Dígito"]).value_counts().reset_index(name="Frecuencia")

# Visualizar las frecuencias con un gráfico de barras
fig_freq = px.bar(digit_counts, x="Dígito", y="Frecuencia", title="Frecuencia de cada dígito en MNIST")
st.plotly_chart(fig_freq)
