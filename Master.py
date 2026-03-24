import streamlit as st

st.set_page_config(
    page_title="Interactive Models for Groundwater Flow and Solute Transport",
    page_icon="Illinois_logo_fullcolor_rgb.png",
) 

col1, col2 = st.columns([1, 4])

with col1:
    st.image("Illinois_logo_fullcolor_rgb.png", width=150)

with col2:
    
    st.markdown(
        "<h1 style='margin-bottom:0px;'>Interactive Models for Groundwater Flow and Solute Transport</h1>",
        unsafe_allow_html=True
    )

st.header('Welcome 👋')
st.markdown(
    """
    This application is developed to illustrate solutions to common problems in solute transport and groundwater flow. 
    
    Cases include solute transport with 1D, 2D, and 3D dispersion, flow and particle tracking in steady flow, generation or random permeability fields.
"""
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")

st.sidebar.markdown(
    """
    
    - [2D Domenico Model](https://2d-sorption.streamlit.app/)
    
    """
)

st.sidebar.info("Select a tool to navigate.")