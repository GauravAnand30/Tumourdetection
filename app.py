import sqlite3
import numpy as np
import streamlit as st
import tensorflow as tf
from sqlite3 import Connection
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Brain Tumour Classifier",
    page_icon="🧠",
    initial_sidebar_state="expanded",
    layout="wide",
)

URI_SQLITE_DB = "predictions.db"

def init_db(conn: Connection):
    conn.execute('CREATE TABLE IF NOT EXISTS userstable(PREDICTION TEXT)')
    conn.commit()

def app():
    interpreter = tf.lite.Interpreter(model_path='brain.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    st.markdown(
        """
        <style>
        body {
            background-color: #222;
            color: white;
        }
        .main-title {
            text-align: center; 
            color: white;
            font-size: 40px;
        }
        .sub-title {
            text-align: center; 
            color: white;
            font-size: 25px;
        }
        .sidebar-title {
            color: white;
            font-size: 20px;
        }
        .sidebar-content {
            color: white;
            font-size: 18px;
        }
        .file-uploader {
            text-align: center;
            font-size: 20px;
            color: white;
        }
        .result {
            text-align: center;
            color: #00FF00;
            font-size: 30px;
        }
        .remedies {
            text-align: center;
            margin-top: 20px;
        }
        .remedies h3 {
            color: #ff6347;
            font-size: 25px;
        }
        .remedies ul {
            list-style-position: inside;
            text-align: left;
            display: inline-block;
            color: white;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h1 class='main-title'>Brain Tumor Classifier 🧠</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-title'>Upload an image to determine the type of tumor</h3>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h2 class='sidebar-title'>Brain Tumors?</h2>", unsafe_allow_html=True)

        with st.expander("About"):
            st.markdown(
                "<p class='sidebar-content'>A brain tumor is a mass or growth of abnormal cells in the brain. Tumors can be either benign (non-cancerous) or malignant (cancerous). They can originate in the brain itself or spread from other parts of the body.</p>",
                unsafe_allow_html=True,
            )

        with st.expander("Symptoms and Signs"):
            st.markdown(
                "<p class='sidebar-content'>Common symptoms of brain tumors include headaches, seizures, changes in vision, difficulty speaking or understanding speech, and changes in mood or personality.</p>",
                unsafe_allow_html=True,
            )

        with st.expander("How to Monitor Brain Health and Seek Help"):
            st.markdown(
                """
                <ul class='sidebar-content'>
                    <li>Regular medical check-ups and screenings can help monitor brain health.</li>
                    <li>Pay attention to any unusual symptoms and seek medical advice if you notice persistent changes.</li>
                    <li>Early detection and treatment are crucial for better outcomes.</li>
                </ul>
                """,
                unsafe_allow_html=True,
            )

    file = st.file_uploader("<h4 class='file-uploader'>Please upload your MRI Scan</h4>", type=["png", "jpg", "jpeg"], accept_multiple_files=False, label_visibility="collapsed")

    conn = get_connection(URI_SQLITE_DB)
    init_db(conn)

    def import_and_predict(image_data):
        size = (256, 256)
        image1 = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        image1 = image1.convert('RGB')
        img = np.array(image1) / 255.0
        img_reshape = img[np.newaxis, ...]

        interpreter.set_tensor(input_details[0]['index'], img_reshape.astype(np.float32))
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])

        return prediction

    labels = ['pituitary', 'notumor', 'glioma', 'meningioma']

    if file is not None:
        image = Image.open(file)
        st.image(image, width=300)
        predictions = import_and_predict(image)
        predictions = np.argmax(predictions)
        prediction_label = labels[predictions]
        result_text = f"<h2 class='result'>The patient most likely has {prediction_label}</h2>"

        st.markdown(result_text, unsafe_allow_html=True)

        if prediction_label != 'notumor':
            remedies = """
            <div class='remedies'>
                <h3>Recommended Actions:</h3>
                <ul>
                    <li>Consult a neuro-oncologist immediately.</li>
                    <li>Follow the treatment plan as advised by your healthcare provider.</li>
                    <li>Consider getting a second opinion for treatment options.</li>
                    <li>Maintain a healthy lifestyle to support overall brain health.</li>
                    <li>Seek support from mental health professionals and support groups.</li>
                </ul>
            </div>
            """
            st.markdown(remedies, unsafe_allow_html=True)

@st.cache_resource
def get_connection(path: str):
    return sqlite3.connect(path, check_same_thread=False)

if __name__ == '__main__':
    app()
