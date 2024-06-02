import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Загрузка модели
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Файл модели не найден. Убедитесь, что model.pkl загружен в репозиторий.")

st.title('Предсказание цены квартиры')

# Ввод параметров квартиры
way = st.selectbox('Способ передвижения до метро', ['walk', 'drive'])
provider = st.selectbox('Поставщик объявления', ['realtor', 'owner'])
views = st.number_input('Количество просмотров', min_value=0)
fee_percent = st.number_input('Комиссия (%)', min_value=0.0, max_value=100.0)
storey = st.number_input('Этаж', min_value=1)
minutes = st.number_input('Время до метро (минут)', min_value=0)
storeys = st.number_input('Общее количество этажей в доме', min_value=1)
living_area = st.number_input('Жилая площадь (кв.м)', min_value=0.0)
kitchen_area = st.number_input('Площадь кухни (кв.м)', min_value=0.0)
total_area = st.number_input('Общая площадь (кв.м)', min_value=0.0)

# Преобразование данных для модели
input_data = pd.DataFrame({
    'views': [views],
    'fee_percent': [fee_percent],
    'storey': [storey],
    'minutes': [minutes],
    'storeys': [storeys],
    'living_area': [living_area],
    'kitchen_area': [kitchen_area],
    'total_area': [total_area],
    'way': [way],
    'provider': [provider]
})

# Применение тех же преобразований, что и при обучении модели
input_data_transformed = model.named_steps['preprocessor'].transform(input_data)

# Предсказание цены
if st.button('Предсказать цену'):
    prediction = model.named_steps['regressor'].predict(input_data_transformed)
    st.write(f'Предсказанная цена: {prediction[0]:,.2f} руб.')
