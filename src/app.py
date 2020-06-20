import streamlit as st
from model import predict

# Minimal code for frontend using streamlit
st.title("Upload An Image to Classify (JPG files only)")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
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
