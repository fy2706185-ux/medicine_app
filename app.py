
import streamlit as st
import pickle
import json

st.title("AI Medicine Recommendation System")

# Load files
with open("disease.json","r") as f:
    data = json.load(f)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

symptom = st.text_input("Enter your symptoms")

if st.button("Recommend"):
    vec = vectorizer.transform([symptom])
    pred = model.predict(vec)[0]
    st.write("Disease:", pred)
    st.write("Medicines:", data[pred]["medicine"])
    st.write("Precautions:", data[pred]["precautions"])
