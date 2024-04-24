import streamlit as st
import sys

sys.path.append('modules')
import uuid
from main import *
import os
import pandas as pd
import pathlib

model=load_model(path='VGG16_cat_dog.h5')

st.set_page_config(
    page_title="Dog vs Cat",
    page_icon="",
)

if 'uuid' not in st.session_state:
    st.session_state['uuid'] = str(uuid.uuid4())

st.title("Dog vs cat")
st.text("Drop a file to know whether it is a cat or a dog :")



uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    if not os.path.exists(f"temp/{st.session_state['uuid']}") : os.mkdir(f"temp/{st.session_state['uuid']}")
    with open(f"temp/{st.session_state['uuid']}/{st.session_state['uuid']}.jpg", 'wb') as f:
        f.write(bytes_data)
        f.close()
    img_width, img_height = 224, 224
    img = image.load_img(f"temp/{st.session_state['uuid']}/{st.session_state['uuid']}.jpg", target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction=predict_class(model=model,image=x)
    if prediction[0][0]<0.5 : st.write("cat")
    else : st.write("dog")