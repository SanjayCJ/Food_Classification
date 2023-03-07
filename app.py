import streamlit as st
import tensorflow as tf
import cv2
import os
from PIL import Image, ImageOps
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
st.set_option('deprecation.showfileUploaderEncoding', False) # to avoid warnings while uploading files

# Here we will use st.cache so that we would load the model only once and store it in the cache memory which will avoid re-loading of model again and again.
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('my_model4.hdf5')
  return model

# load and store the model
with st.spinner('Model is being loaded..'):
  model=load_model()
  
# Function for prediction
def import_and_predict(image_data, model):
    size = (128,128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction
def main():
    st.title("Food Image Classifier")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Food Image Classifier App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])
    class_names=['Bread','Soup','Vegetable-Fruit']
    result=""
    final_images=""
    with st.sidebar:
      with st.expander("Upload an image of one of these food"):
        
#       st.header("Please upload an image from one of these categories")
        st.text("1. Bread")
        st.text("2. Soup")
        st.text("3. Vegetable-Fruit")
      st.headerFood Classifier using VGG16")
      st.image("vgg16.jpg")
            
    if st.button("Predict"):
      if file is None:
        st.write("please upload an image")
      else:
        image = Image.open(file)
        
        predictions = import_and_predict(image,model)
        score = tf.nn.softmax(predictions[0])
        result= class_names[np.argmax(predictions[0])]
#         st.write('This is {} '.format(result))
        html_temp = f"""
                    <div style="background-color:tomato;padding:10px">
                    <h2 style="color:white;text-align:center;"> This is {result} </h2>
                    </div>
                     """
        st.markdown(html_temp,unsafe_allow_html=True)
        st.image(image, use_column_width=True)
      
      
        st.caption("The result is trained on similar images like: ")
        
        train_path_bread=[]
        train_path_soup=[]
        train_path_vegetable_Fruit=[]
        
        for folder_name in ['Bread/','Soup/','Vegetable_Fruit/']:
    
          #Path of the folder
          images_path = os.listdir(folder_name)

          for i, image_name in enumerate(images_path): 
            if folder_name=='Bread/':
                train_path_bread.append(folder_name+image_name)
            elif folder_name=='Soup/':
                train_path_soup.append(folder_name+image_name)
            elif folder_name=='Vegetable_Fruit/':
                train_path_vegetable_Fruit.append(folder_name+image_name)

        Bread=[]
        Soup=[]
        Vegetable_Fruit=[]
        
        for i in train_path_bread:
          image = Image.open(i).resize((200, 200))
          Bread.append(image)
        for i in train_path_soup:
          image = Image.open(i).resize((200, 200))
          Soup.append(image)
        for i in train_path_vegetable_Fruit:
          image = Image.open(i).resize((200, 200))
          Vegetable_Fruit.append(image)
        


        if result=='Bread':
            final_images =Bread

        elif result=='Soup':
            final_images =Soup

        elif result=='Vegetable-Fruit':
            final_images =Vegetable_Fruit


        n_rows = 1 + len(final_images) // int(4)
        rows = [st.container() for _ in range(n_rows)]
        cols_per_row = [r.columns(4) for r in rows]
        cols = [column for row in cols_per_row for column in row]

        for image_index, mon_image in enumerate(final_images):
            cols[image_index].image(mon_image)

    if st.button("About"):
     st.text("This is a food classifier that predicts the food of the image uploaded for 3 different monuments (list provided on the left).")
     st.text("This classifier uses VGG16, a pre-trained Convolutional Neural Network architecture.")
     st.text("This classifier has been deployed using Streamlit.")
if __name__=='__main__':
    main()
