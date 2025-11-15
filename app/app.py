import streamlit as st

pages = {
    "Главная": [
        st.Page("main.py", title="Анализировать")
    ],
    "Статистика": [
        st.Page("stats.py", title="Данные")
    ],
    "Информация": [
        st.Page("info.py", title="О приложении"),
        st.Page("developer.py", title="Контакты")
    ],
}

pg = st.navigation(pages)
pg.run()

