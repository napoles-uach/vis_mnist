import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import random
import pandas as pd
import matplotlib.pyplot as plt

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

#############


# Función para mostrar la matriz 5x5 de imágenes aleatorias
def show_random_images():
    # Crear una figura para la matriz 5x5
    fig, axes = plt.subplots(4, 5, figsize=(5,5))
    
    # Muestreo aleatorio de imágenes
    random_indices = np.random.choice(len(X_train), 20, replace=False)
    
    # Iterar sobre la cuadrícula 5x5 y mostrar las imágenes aleatorias
    for i, ax in enumerate(axes.flat):
        # Seleccionar una imagen aleatoria
        random_image = X_train[random_indices[i]]
        random_label = random_indices[i]#y_train[random_indices[i]]
        
        # Mostrar la imagen en la cuadrícula
        ax.imshow(random_image, cmap='gray')
        ax.set_title(f'index:{random_label}', fontsize=8)
        ax.axis('off')  # Ocultar los ejes
    
    # Ajustar el espaciado entre los subgráficos
    plt.tight_layout()
    st.pyplot(fig)

# Título de la aplicación
st.title('Matriz de 5x5 Imágenes de MNIST')

# Botón para actualizar la muestra aleatoria
if st.button('Generar nueva muestra aleatoria'):
    show_random_images()
else:
    show_random_images()


###################33
# Cargar los datos de MNIST
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Dividir el conjunto de entrenamiento completo en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Crear un DataFrame para almacenar las etiquetas y los conjuntos correspondientes
data = {
    'Conjunto': ['Entrenamiento'] * len(y_train) + ['Validación'] * len(y_val) + ['Prueba'] * len(y_test),
    'Dígito': np.concatenate([y_train, y_val, y_test])
}

df = pd.DataFrame(data)

# Crear la gráfica de barras utilizando Plotly
fig = px.histogram(df, x='Dígito', color='Conjunto', barmode='group',
                   title='Distribución de los dígitos en los conjuntos de entrenamiento, validación y prueba',
                   labels={'Dígito': 'Dígito', 'count': 'Frecuencia'},
                   category_orders={"Conjunto": ["Entrenamiento", "Validación", "Prueba"]})

# Mostrar la gráfica en Streamlit
st.title('Distribución de los dígitos en los conjuntos de entrenamiento, validación y prueba')
st.plotly_chart(fig, use_container_width=True)


########################3
# Calcular el promedio de las imágenes para cada dígito
mean_images = [np.mean(X_train[y_train == i], axis=0) for i in range(10)]

# Crear una figura para mostrar los mapas de calor
fig, ax = plt.subplots(2, 5, figsize=(10, 5))

# Mostrar los mapas de calor para las imágenes promedio de cada dígito usando solo Matplotlib
for i in range(10):
    ax[i // 5, i % 5].imshow(mean_images[i], cmap='viridis')
    ax[i // 5, i % 5].set_title(f'Dígito {i}')
    ax[i // 5, i % 5].axis('off')

# Ajustar el espaciado
plt.tight_layout()

# Mostrar la figura en Streamlit
st.title('Mapas de calor de las imágenes promedio por dígito en MNIST')
st.pyplot(fig)
