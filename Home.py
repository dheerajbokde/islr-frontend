import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="ISLR",
    page_icon="👋",
)

st.sidebar.success("Select an option above.")
st.markdown('<div style="text-align: center; font-size: 18px; ">Welcome to</div>', unsafe_allow_html=True)
st.markdown('')
st.markdown('<div style="text-align: center; font-size: 25px; ">Capstone Project</div>', unsafe_allow_html=True)
st.markdown('')
logo = Image.open('artifacts/images/iisc-logo.png')
st.image(logo, use_column_width=True)
st.markdown('<div style="text-align: center; font-size: 20px; ">Advanced Programme in Computational Data Science</div>', unsafe_allow_html=True)
st.markdown('')
st.markdown('<div style="text-align: center; font-size: 18px; ">on</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; font: bold; font-size: 40px; "> <em><strong>Generate English Text From Indian Sign Language (ISL) using Deep Learning</strong></em></div>', unsafe_allow_html=True)
st.markdown('')
img = Image.open('artifacts/images/isl.png')
st.image(img, use_column_width=True)
st.markdown('')
st.markdown('<div style="text-align: center; font-size: 18px; ">Presented By</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; font-size: 30px; ">Group 6</div>', unsafe_allow_html=True)

col1,col2,col3,col4 = st.columns(4)
name_1 = ['Aravind Durgoji','Dheeraj Bokde','Divya Abraham','Himanshu Singh']
name_2 = ['Indranil Saha','Sumit Aggarwal','Tarun Julasaria']
with col2:
    for i in name_1:
        name1_html_str = f"""
            <style>
            p.a {{
                font: Bold 18px Courier;
            }}
            </style>
            <p class="a">{i}</p>
        """
        st.markdown(name1_html_str, unsafe_allow_html=True)

with col3:
    for i in name_2:
        name2_html_str = f"""
            <style>
            p.a {{
                font: Bold 18px Courier;
            }}
            </style>
            <p class="a">{i}</p>
        """
        st.markdown(name2_html_str, unsafe_allow_html=True)

st.markdown('')
st.markdown('<div style="text-align: center; font-size: 18px; ">Mentored By</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; font: Bold 20px Courier; ">Piyush Pathak</div>', unsafe_allow_html=True)
