import streamlit as st
import os
import cv2
import numpy as np
import random
from dataset import LoadDataSet

# Charger le dataset
st.title("Visualisation du Dataset de Formes")
dataset = LoadDataSet()

# Liste des types d'images disponibles
types = sorted(list(set(dataset.Types())))
types.insert(0, "Toutes")

# Sélection du type d'image
selected_type = st.selectbox("Choisissez un type de forme :", types)

# Filtrer les images
if selected_type == "Toutes":
    images = dataset.Images()
else:
    images = dataset[selected_type]

# Afficher un nombre limité d'images aléatoires
num_images = min(len(images), 10)
random_images = random.sample(images, num_images)

st.write(f"Affichage de {num_images} images")
cols = st.columns(5)
for i, img in enumerate(random_images):
    with cols[i % 5]:
        st.image(img, use_column_width=True, channels="GRAY")

