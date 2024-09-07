import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Cargar el conjunto de datos MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

st.title('Capacidades de Visualización con el Conjunto MNIST')

# Sidebar para seleccionar las visualizaciones
st.sidebar.header('Opciones de Visualización')

# 1. Visualización de múltiples imágenes simultáneamente
st.header('Visualización de Múltiples Imágenes')
cols = st.columns(5)  # Mostrar 5 imágenes por fila
for i in range(5):
    index = np.random.randint(0, len(x_train))
    with cols[i]:
        st.image(x_train[index], caption=f'Etiqueta: {y_train[index]}', use_column_width=True)

# 2. Mostrar la matriz de confusión con gráfico de calor
st.header('Matriz de Confusión (Gráfico de Calor)')
if st.sidebar.checkbox('Mostrar matriz de confusión'):
    # Entrenar un modelo simple para predicciones (solo para fines de visualización)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, verbose=0)  # Entrenamiento rápido
    y_pred = np.argmax(model.predict(x_test), axis=1)

    # Generar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Etiqueta Verdadera')
    st.pyplot(fig)

# 3. Mostrar histograma de intensidades de píxeles
st.header('Histograma de Intensidades de Píxeles')
if st.sidebar.checkbox('Mostrar histograma de intensidades'):
    index = st.sidebar.slider('Selecciona un índice de imagen', 0, len(x_train)-1, 0)
    image = x_train[index]
    fig, ax = plt.subplots()
    ax.hist(image.ravel(), bins=256, color='gray', alpha=0.7)
    ax.set_title(f'Histograma de Intensidades para la Imagen {index}')
    st.pyplot(fig)

# 4. Reducción de dimensionalidad y gráfico 3D interactivo con PCA
st.header('Reducción de Dimensionalidad con PCA (Gráfico 3D)')
if st.sidebar.checkbox('Mostrar gráfico 3D interactivo (PCA)'):
    pca = PCA(n_components=3)
    x_train_pca = pca.fit_transform(x_train.reshape(-1, 28*28))

    fig = go.Figure(data=[go.Scatter3d(
        x=x_train_pca[:, 0],
        y=x_train_pca[:, 1],
        z=x_train_pca[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=y_train,
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    fig.update_layout(title='Visualización PCA 3D de MNIST', margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig)

# 5. Zoom interactivo sobre las imágenes
st.header('Zoom Interactivo sobre una Imagen')
if st.sidebar.checkbox('Mostrar zoom interactivo'):
    index = st.sidebar.slider('Selecciona una imagen para hacer zoom', 0, len(x_train)-1, 0)
    image = x_train[index]

    fig = px.imshow(image, color_continuous_scale='gray')
    fig.update_layout(coloraxis_showscale=False, title=f'Zoom interactivo en imagen de dígito {y_train[index]}')
    st.plotly_chart(fig)

