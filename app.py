import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import keras


st.write("#### Esta es una aplicacion que clasifica celulas")

df = pd.read_csv("cell_samples.csv")

atributos = ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BlandChrom', 'NormNucl', 'Mit']


columnas_x = df[atributos]

x = columnas_x.values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x)


imagen_celula = Image.open('imagenes/cell.jpg')
st.image(imagen_celula)




col1, col2, col3 = st.columns((2,1,2))

x_usuario = np.zeros(len(atributos))

with col1:

    for i in range(len(atributos)):
        x_usuario[i] = st.number_input(atributos[i],step=1)
 

with col3:
    Red_neuronal = keras.models.load_model('cell_model.h5')
    x = x_usuario.reshape(1,-1)
    x2 = scaler.transform(x)

    y_pred = Red_neuronal.predict(x2)

    if y_pred < 0.5:
        y_pred = 0
    else:
        y_pred = 1

    st.write("Clase de la celula = ", y_pred) 

    if y_pred == 0:
        im = Image.open("imagenes/0.jpeg")
        st.image(im)
    if y_pred == 1:
        im = Image.open("imagenes/1.jpeg")
        st.image(im)
