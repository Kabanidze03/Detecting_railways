import time
import torch.nn as nn
import streamlit as st
from PIL import Image
from Utils import image_transform, download_model, feature_extraction, download_classifier, post_processing_image, make_probability_diagram

# Инициализируем нашу модель, которую мы будем использовать для извлечения признаков
model = download_model()
# Инициализируем наш классификатор, который был заранее зафайнтюнен на нашу задачу
classifier = download_classifier()


st.header(":blue[Обнаружение железных дорог на фотографиях со спутников🚂]", 
            divider='grey',
            width='content'
)

st.subheader(":grey[Добро пожаловать в наш сервис, загрузите изображение ниже]")

uploaded_file = st.file_uploader(label=":grey[**Загрузите изображение**]", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Изображение пользователя", use_container_width=True)
    image = Image.open(uploaded_file).convert('RGB')
    img_tensor = image_transform(image)

    # Прогоням изображение через модель и получаем вектор
    progress_text = ":grey[Прогоняем изображение через модель и классификатор]"
    my_bar = st.progress(0, text=progress_text)

    for i in range(100):
        time.sleep(0.01)  # Имитируем время обработки
        my_bar.progress(i + 1, text=progress_text)

    out_from_nn = feature_extraction(img_tensor=img_tensor, model=model)
    out_from_nn = post_processing_image(out_from_nn)
    out = classifier.predict_proba(out_from_nn)
    st.success("Готово!")

    st.subheader(":grey[Результаты предсказания:]", divider='grey')
    make_probability_diagram(out)
    st.write(':grey[Так как используется не очень продвинутая модель, результат выдается как вероятности того, есть ли ж/д или нет]')
    
    

    

    
    
