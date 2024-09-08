import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px

# Cargar el conjunto de datos MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

st.title('Red Neuronal con Visualización de Capas Internas (MNIST)')

# Sidebar para configuración del modelo
st.sidebar.header('Configuración de la Red Neuronal')
num_epochs = st.sidebar.slider('Número de épocas', 1, 10, 3)
layer_selection = st.sidebar.selectbox('Selecciona la capa a visualizar', ['Capa 1', 'Capa 2', 'Capa 3'])

# Normalizar los datos
x_train = x_train / 255.0
x_test = x_test / 255.0

@st.cache_resource
def entrenar_modelo(num_epochs):
    # Definir la red neuronal
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name="Input"),
        tf.keras.layers.Dense(128, activation='relu', name="Hidden_Layer_1"),
        tf.keras.layers.Dense(64, activation='relu', name="Hidden_Layer_2"),
        tf.keras.layers.Dense(10, activation='softmax', name="Output_Layer")
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Entrenar el modelo
    history = model.fit(x_train, y_train, epochs=num_epochs, verbose=0)
    return model, history

# Entrenar el modelo solo si cambian los parámetros
st.write(f"Entrenando el modelo con {num_epochs} épocas...")
model, history = entrenar_modelo(num_epochs)

# Visualizar la precisión del modelo
st.header('Precisión del Modelo')
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='Precisión de entrenamiento')
ax.set_xlabel('Época')
ax.set_ylabel('Precisión')
ax.legend()
st.pyplot(fig)

# Realizar una predicción para asegurarse de que el modelo haya sido llamado
model.predict(np.expand_dims(x_test[0], axis=0))

# Función para obtener activaciones de capas
def get_activations(model, layer_name, input_data):
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(input_data)

# Seleccionar imagen para visualizar activaciones
st.header('Visualización de Activaciones de Capas Internas')
index = st.sidebar.slider('Selecciona un índice de imagen para ver activaciones', 0, len(x_test)-1, 0)
selected_image = np.expand_dims(x_test[index], axis=0)

# Mostrar la imagen seleccionada
st.image(x_test[index], caption=f'Imagen del dígito: {y_test[index]}', width=150)

# Obtener y visualizar las activaciones de la capa seleccionada
layer_name = "Hidden_Layer_1" if layer_selection == "Capa 1" else ("Hidden_Layer_2" if layer_selection == "Capa 2" else "Output_Layer")
activations = get_activations(model, layer_name, selected_image)

st.write(f"Activaciones de la {layer_selection}")
if len(activations.shape) == 2:  # Si es una capa densa
    fig, ax = plt.subplots()
    sns.heatmap(activations, cmap="viridis", ax=ax)
    ax.set_title(f"Activaciones de la {layer_selection}")
    st.pyplot(fig)
else:  # Para futuras capas convolucionales
    num_filters = activations.shape[-1]
    fig, axs = plt.subplots(1, num_filters, figsize=(20, 3))
    for i in range(num_filters):
        axs[i].imshow(activations[0, :, :, i], cmap='viridis')
        axs[i].axis('off')
    st.pyplot(fig)

# Realizar predicción
pred = model.predict(selected_image)
st.write(f"Predicción del modelo: {np.argmax(pred)} (Etiqueta verdadera: {y_test[index]})")
