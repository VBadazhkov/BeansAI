import psycopg2
import streamlit as st
import pandas as pd
import os
from common import prediction_sample, CoffeeCNN, DataBaseConnection, get_connection
from PIL import Image

connection = get_connection()

st.title("AI определение обжарки кофе по фото")

st.write("Внимание! Это демо-версия приложения. Загруженные данные будут доступны всем пользователям")

st.header("Загрузите фото")
uploaded_file = st.file_uploader(label="Загрузка", type=["jpg", "png"], label_visibility="hidden")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    if st.button("Анализировать"):
        try:
            class_of_roast, probability = prediction_sample(image)
            st.success(f"Обжарка {class_of_roast} с вероятностью {probability}")
            mark = st.feedback("stars")

            with DataBaseConnection(connection) as cursor:
                query = "INSERT INTO statistics (time, path, class, prob, feedback) VALUES (NOW(), %s, %s, %s, %s)"
                cursor.execute(query, (uploaded_file.name, class_of_roast, probability, mark))

        except Exception as e:
            st.write(f"Ошибка обработки: {e}")
