import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

st.title('Предсказание цены квартиры')
area = st.number_input('Площадь квартиры (кв.м)')
floor = st.number_input('Этаж')
total_floors = st.number_input('Общее количество этажей в доме')
distance_to_metro = st.number_input('Расстояние до метро (минут)')

input_data = np.array([[area, floor, total_floors, distance_to_metro]])

if st.button('Предсказать цену'):
    prediction = model.predict(input_data)
    st.write(f'Предсказанная цена: {prediction[0]:,.2f} руб.')
