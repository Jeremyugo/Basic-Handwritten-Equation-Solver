from functions import Process_image, Calculate
import numpy as np
import streamlit as st
import keras
from keras.models import load_model


# dictionary for class labels
map_dict = {0: '-',
 1: '+',
 2: 'div',
 3: 'X',
 4: 'times',
 5: '0',
 6: '1',
 7: '2',
 8: '3',
 9: '4',
 10: '5',
 11: '6',
 12: '7',
 13: '8',
 14: '9'}



st.title("Handwritten Equation Solver")
st.markdown("by: Jeremiah Chinyelugo")

st.write(" ")
st.write(" ")

#"=============================================================================="

# Introduction
st.write("""
         This is a simple app that can solve basic handwritten mathematical equations.
         Either upload an image of your equation or take a picture of it through this app to get a solution to your equation.
         Your image will be fed through an image processing function, then passed to the Convolutional Neural Network in real time. 
         The solution to your equation will then be calculated based on model predictions using a custom function.""")

st.write(" ")
st.write(" ")

#"=============================================================================="

# Visualizing the sample Images data was trained on
with st.expander("Disclaimer"):
    st.write("""
             1. This model can only solve basic equations with not more than 2 math operators.
             2. A Sample of each symbol used in training the model can be found below, and handwritten equations should mirror these symbols to get the correct solution to the equation.
             """)
    col1_a,col2_a,col3_a,col4_a,col5_a,col6_a,col7_a = st.columns(7)
    col1_b,col2_b,col3_b,col4_b,col5_b,col6_b,col7_b = st.columns(7)
    
    # first row
    col1_a.image("./sample_images/0.jpg", caption="0",
             width=80)
    col2_a.image("./sample_images/1.jpg", caption="1",
             width=80)
    col3_a.image("./sample_images/2.jpg", caption="2",
             width=80)
    col4_a.image("./sample_images/3.jpg", caption="3",
             width=80)
    col5_a.image("./sample_images/4.jpg", caption="4",
             width=80)
    col6_a.image("./sample_images/5.jpg", caption="5",
             width=80)
    col7_a.image("./sample_images/6.jpg", caption="6",
             width=80)
    
    # second row
    col1_b.image("./sample_images/7.jpg", caption="7",
             width=80)
    col2_b.image("./sample_images/8.jpg", caption="8",
             width=80)
    col3_b.image("./sample_images/9.jpg", caption="9",
             width=80)
    col4_b.image("./sample_images/+.jpg", caption="+",
             width=80)
    col5_b.image("./sample_images/-.jpg", caption="-",
             width=80)
    col6_b.image("./sample_images/times.jpg", caption="times",
             width=80)
    col7_b.image("./sample_images/div.jpg", caption="div",
             width=80)

#"=============================================================================="
    
st.write(" ")
st.write(" ")    

# loading keras model
model = load_model("Equation_Solver_Model_3.h5")  

#"=============================================================================="

# options for uploading image

options = st.radio("Choose your input method",
                   ("Upload Image", "Use Camera"))


#"=============================================================================="

# processing the image and solving the equation

if options == "Upload Image":
    file = st.file_uploader("Upload your handwritten equation (Please ensure the image is clear, and handwritting, legible)", ['jpg','png', 'jfif'], accept_multiple_files=False)
    
    if file:
        st.image(file, caption="Image you uploaded")
        # a. processing the image
        
        segmented_images = Process_image(file)
        if np.sum(segmented_images) < 100:
            st.write("Please check your image and following the instructions")
        else:
            # b. model prediction
            pred = np.argmax(model.predict(segmented_images), axis=1)
            
            # c. performing the calculation
            equation_list = []
            for array in pred:
                equation_list.append(map_dict[array])
            answer, equation = Calculate(equation_list)
            
            if answer == None:
                st.write("Please check your image and following the instructions")
            else:
                st.write(f"Handwritten Equation Solution: {equation}  =  {answer}")
                
                

if options == "Use Camera":
    file = st.camera_input("Take a picture of the equation (Please ensure the image is clear)")
    
    if file is None:
        pass
        
    else:
        segmented_images = Process_image(file)
        st.write(segmented_images.shape)
        if np.sum(segmented_images) < 100:
            st.write("Please check your image and following the instructions")
        
        else:
            # b. model prediction
            pred = np.argmax(model.predict(segmented_images), axis=1)
            st.write(pred)
            
            # c. performing the calculation
            equation_list = []
            for array in pred:
                equation_list.append(map_dict[array])
            answer, equation = Calculate(equation_list)
            
            if answer == None:
                st.write("Please check your image and following the instructions")
            else:
                st.write(f"Handwritten Equation Solution: {equation}  =  {answer}")
            
    
