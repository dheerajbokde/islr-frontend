import streamlit as st
from itertools import cycle

contacts = [
  {
    'name': 'Aravind Durgoji',
    'image': 'artifacts/images/members/img.png',
  },
  {
    'name': 'Dheeraj Bokde',
    'image': 'artifacts/images/members/img.png',
  },
  {
    'name': 'Divya Abraham',
    'image': 'artifacts/images/members/img.png',
  },
  {
    'name': 'Himanshu Singh',
    'image': 'artifacts/images/members/img.png',
  },
  {
    'name': 'Indranil Saha',
    'image': 'artifacts/images/members/img.png',
  },
  {
    'name': 'Sumit Aggarwal',
    'image': 'artifacts/images/members/img.png',
  },
  {
    'name': 'Tarun Julasaria',
    'image': 'artifacts/images/members/img.png',
  }
]


st.set_page_config(layout="centered")
st.title(':blue[Acknowledment]')
st.markdown(
    """
    ### Thanks to IISc Professors and TalentSprint Team
    - Prof. Sashikumaar Ganesan
    - Prof. Deepak Subramani
    - Prof. Sundeep Prabhakar Chepuri
    - Prof. Yogesh Simmhan
    - Prof. Shashi Jain
    - Dr. Surobhi and Team

    ### Thanks to Mentor
    - Piyush Pathak

    ### Contact Us - Group 6
"""
)
image_list = []
caption = []
for i in contacts:
  image_list.append(i['image'])
  caption.append(i['name'])

cols = cycle(st.columns(4))
for idx, image_list in enumerate(image_list):
    next(cols).image(image_list, width=150, caption=caption[idx])


