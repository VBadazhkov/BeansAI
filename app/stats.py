import streamlit as st
import psycopg2
import pandas as pd
import os
from common import DataBaseConnection, get_connection

st.title("Результаты")

if st.button("Очистить данные", use_container_width=True):
    query = "TRUNCATE TABLE statistics;"

    connection = get_connection()
    with DataBaseConnection(connection) as cursor:
        cursor.execute(query)

connection = get_connection()
query = "SELECT * FROM statistics"

with DataBaseConnection(connection) as cursor:
    cursor.execute(query)
    columns = [data[0] for data in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data=data, columns=columns)
    st.dataframe(df)


