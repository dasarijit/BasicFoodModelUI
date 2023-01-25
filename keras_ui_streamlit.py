# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:02:34 2021

@author: User
"""

import predict_streamlit as app1
import streamlit as st

# Define pages based on apps imported.
PAGES = {
    "Predict": app1
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
