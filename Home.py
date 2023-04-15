import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="ISLR",
    page_icon="👋",
)

st.write("### 🙏 Welcome to")
st.write("# :blue[I]ndian :blue[S]ign :blue[L]anguage :blue[R]ecognition")
img = Image.open('artifacts/images/WELCOME.jpg')
st.image(img, use_column_width=True)
st.sidebar.success("Select an option above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)