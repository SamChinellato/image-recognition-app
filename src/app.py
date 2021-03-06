import streamlit as st
from model import predict

# Minimal code for frontend using streamlit
st.title("Upload An Image to Classify")
uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
    try:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        img = uploaded_file.read()
        # pass uploaded image to model
        try:
            result = predict(img)
            # Result output
            st.write('%s (%.2f%%)' % (result[0], result[1] * 100))
        except Exception as e:
            st.write("Something went wrong, unable to classify image")
            st.write("For debug: {}".format(e))
    except Exception as e:
        st.write("Couldn't read that file. Please try another format")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("Made by Sam Chinellato. Deployed using AWS. [View the App sourcecode here]("
            "https://github.com/SamChinellato/image-recognition-app)")

