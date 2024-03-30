import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np 

# Define function for image classification
def classify_image(model, image, class_names):
    # Perform image classification using the loaded model
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    return predicted_class

# Define function to load DenseNet201 model
def load_densenet201_model(model_path, num_classes):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        input_shape = (224, 224, 3)  # Specify the input shape
        model.build(input_shape)
        
        # Ensure the model output matches the number of classes
        assert model.output_shape[1] == num_classes, f"Model output shape mismatch. Expected {num_classes} classes."
        
        return model
    except Exception as e:
        st.error(f"Error loading DenseNet201 model: {str(e)}")
        return None

# Define Streamlit app layout
st.title("Deep Learning Model Selection")

# Define buttons for each model
model_names = ["All","Cervix","Lung","Oral","Kidney","Breast","Brain","lymph"]
selected_model = st.radio("Select Model:", model_names)

# Add file uploader for image input
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Display the selected image
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
class_names =  ['all_benign', 'all_early', 'all_pre', 'all_pro']
class_names = {
    'All': ['all_benign', 'all_early', 'all_pre', 'all_pro'],  # Replace class1, class2, class3 with actual class names for Brain model
    'Cervix':  ['cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi'],
    'Lung': ['colon_aca', 'colon_bnt', 'lung_aca', 'lung_bnt', 'lung_scc'],
    'Oral':['oral_normal', 'oral_scc'],
    'Kidney': ['kidney_normal', 'kidney_tumor'],
    'Breast':['breast_benign', 'breast_malignant'],
    'Brain':  ['brain_glioma', 'brain_menin', 'brain_tumor'],
    'lymph':  ['lymph_cll', 'lymph_fl', 'lymph_mcl'],

}
# Add button to trigger model loading
if st.button("Load Model"):
    if selected_model == "All":
        model_path = "D:\SIG\FInal Year Project\Codes, Simulations and GUI\.h5 files\DenseNet201\ALL  - DenseNet201.h5"
        num_classes = 4  # Update the number of classes
        model = load_densenet201_model(model_path, num_classes)
        if model is not None:
            st.success("All model loaded successfully.")
    elif selected_model == "Cervix":
        model_path = "D:\SIG\FInal Year Project\Codes, Simulations and GUI\.h5 files\DenseNet201\Cervical Cancer  - DenseNet201.h5"
        num_classes = 5 # Update the number of classes
        model = load_densenet201_model(model_path, num_classes)
        if model is not None:
            st.success("Cervix model loaded successfully.")
    elif selected_model == "Lung":
        model_path = "D:\SIG\FInal Year Project\Codes, Simulations and GUI\.h5 files\DenseNet201\Lung and Colon Cancer  - DenseNet201.h5"
        num_classes = 5 # Update the number of classes
        model = load_densenet201_model(model_path, num_classes)
        if model is not None:
            st.success("Lung model loaded successfully.")
    elif selected_model == "Oral":
        model_path = "D:\SIG\FInal Year Project\Codes, Simulations and GUI\.h5 files\DenseNet201\Oral Cancer  - DenseNet201.h5"
        num_classes = 2 # Update the number of classes
        model = load_densenet201_model(model_path, num_classes)
        if model is not None:
            st.success("Oral model loaded successfully.")
    elif selected_model == "Kidney":
        model_path = "D:\SIG\FInal Year Project\Codes, Simulations and GUI\.h5 files\DenseNet201\Kidney Cancer  - DenseNet201.h5"
        num_classes = 2# Update the number of classes
        model = load_densenet201_model(model_path, num_classes)
        if model is not None:
            st.success("Kidney model loaded successfully.")
    elif selected_model == "Breast":
        model_path = "D:\SIG\FInal Year Project\Codes, Simulations and GUI\.h5 files\DenseNet201\Breast Cancer  - DenseNet201.h5"
        num_classes = 2 # Update the number of classes
        model = load_densenet201_model(model_path, num_classes)
        if model is not None:
            st.success("Breast model loaded successfully.")
    elif selected_model == "Brain":
        model_path = "D:\SIG\FInal Year Project\Codes, Simulations and GUI\.h5 files\DenseNet201\Brain Cancer  - DenseNet201.h5"
        num_classes = 3 # Update the number of classes
        model = load_densenet201_model(model_path, num_classes)
        if model is not None:
            st.success("Brain model loaded successfully.")
    elif selected_model == "lymph":
        model_path = "D:\SIG\FInal Year Project\Codes, Simulations and GUI\.h5 files\DenseNet201\Lymphoma  - DenseNet201.h5"
        num_classes = 3 # Update the number of classes
        model = load_densenet201_model(model_path, num_classes)
        if model is not None:
            st.success("Lymph model loaded successfully.")
  
    else:
        st.error("Invalid model selected.")

    # Process the uploaded image and make a prediction
    if model is not None:
        try:
            if uploaded_file is not None:
                # Open the uploaded image
                image = Image.open(uploaded_file)
                if image is not None:
                    # Convert image to numpy array
                    image_array = np.array(image)
                    
                    # Check if the file is an image
                    if image.format not in ["JPEG", "PNG"]:
                        st.error("Uploaded file is not a valid image.")
                    else:
                        # Ensure image has three color channels
                        if len(image_array.shape) == 2:
                            image_array = np.stack((image_array,) * 3, axis=-1)
                        
                        # Resize image to match model input shape (e.g., 224x224 for DenseNet201)
                        image_array = tf.image.resize(image_array, (224, 224))
                        
                        # Expand dimensions to match model input shape
                        image_array = np.expand_dims(image_array, axis=0)
                        
                        # Perform image classification
                        prediction = classify_image(model, image_array,class_names[selected_model])
                        
                        # Display the prediction result
                        st.write("Prediction:", prediction)
                else:
                    st.error("Error: Uploaded file is empty or not a valid image.")
        except Exception as e:
            st.error(f"Error processing the uploaded image: {str(e)}")
