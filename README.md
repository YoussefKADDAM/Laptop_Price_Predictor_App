# 💻 Laptop Price Predictor

A machine learning web application that predicts laptop prices from their hardware specifications.

Built with **Python, Streamlit, Docker, and Azure App Service**.

---

## 📁 Project Structure

app.py # Streamlit web app

Laptop_Prediction_Project.ipynb # ML training notebook

pipe.pkl # Trained ML model

df.pkl # Preprocessed data for the app

laptop_price.csv # Dataset

Dockerfile # Docker container config

requirements.txt # Dependencies

README.md

---

## ▶️ Run Locally

pip install -r requirements.txt
streamlit run app.py

App runs on:
👉 http://localhost:8501/

---

## 🐳 Docker

Build:

docker build -t laptop-price-predictor .

Run:

docker run -p 8501:8501 laptop-price-predictor

---

## ☁️ Azure Deployment

Create Web App → Container → Docker Hub

Use image:

kaddamyoussef/laptop-price-predictor:latest

Add App Setting:

WEBSITES_PORT = 8501

---

## 🛠 Tech Stack

Python, Pandas, NumPy

Scikit-Learn, XGBoost

Streamlit

Docker

Azure App Service

---

## 👤 Author

GitHub: https://github.com/kaddamyoussef

