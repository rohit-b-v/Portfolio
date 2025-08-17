import streamlit as st
import create_map


create_map.create_documents()
nav = st.navigation(pages=['Home.py', 'About Me.py'], position='top')
nav.run()





