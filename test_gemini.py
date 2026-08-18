import streamlit as st
from google import genai

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Say hello to Saniya and tell her that Gemini is connected successfully."
)

st.write(response.text)
