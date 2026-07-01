# 🌱 Smart Crop Recommendation System

> 🎓 Academic Group Project  
> Developed by students of the Department of Computer Science & Engineering (Data Science), St. Thomas College of Engineering and Technology, Mattanur.

## 📖 Overview

The **Smart Crop Recommendation System** is an AI-powered web application that recommends the most suitable crop based on soil nutrient composition and environmental conditions. The system uses a **Stacking Ensemble Machine Learning Model** that combines multiple classifiers to improve prediction accuracy and provide reliable crop recommendations.

The application is built using **Python** and **Streamlit**, providing an interactive interface for users to input agricultural parameters and receive intelligent crop suggestions.

---

## ✨ Features

- 🌾 Intelligent Crop Recommendation
- 🤖 Stacking Ensemble Learning
- 📊 Probability-Based Predictions
- 🔍 Explainable AI using SHAP
- 📈 Model Performance Visualization
- 🎨 Interactive Streamlit Dashboard
- 💾 Automatic Model Saving and Loading

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-Learn |
| Ensemble Learning | Stacking Classifier |
| Gradient Boosting | XGBoost |
| Explainable AI | SHAP |
| Data Processing | Pandas, NumPy |
| Model Serialization | Pickle |

---

## 🧠 Machine Learning Model

The crop recommendation model uses a **Stacking Ensemble Classifier** consisting of:

### Base Models

- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost Classifier

### Meta Model

- Logistic Regression

The outputs from the base models are combined using Logistic Regression to generate the final crop recommendation.

---

## 📊 Input Parameters

The model predicts the most suitable crop using the following parameters:

| Parameter | Description |
|-----------|-------------|
| Nitrogen (N) | Nitrogen content in soil |
| Phosphorus (P) | Phosphorus content in soil |
| Potassium (K) | Potassium content in soil |
| Temperature | Temperature (°C) |
| Humidity | Relative Humidity (%) |
| pH | Soil pH |
| Rainfall | Rainfall (mm) |

---

## 🌱 Supported Crops

The system can recommend crops including:

- Rice
- Maize
- Chickpea
- Kidney Beans
- Pigeon Peas
- Moth Beans
- Mung Bean
- Blackgram
- Lentil
- Pomegranate
- Banana
- Mango
- Grapes
- Watermelon
- Muskmelon
- Apple
- Orange
- Papaya
- Coconut
- Cotton
- Jute
- Coffee

---

## 📂 Project Structure

```text
Smart-Crop-Recommendation/
│
├── app.py
├── model.py
├── Crop_recommendation.csv
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/smart-crop-recommendation.git

cd smart-crop-recommendation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

> **Important:** Run `model.py` before launching `app.py`. The `model.py` script trains the machine learning model and generates the required model files used by the Streamlit application.

## Step 1: Train the Model

```bash
python model.py
```

This will:

- Load and preprocess the dataset
- Train the Stacking Ensemble Model
- Save the trained model files

Wait until the training process completes successfully before proceeding.

---

## Step 2: Launch the Streamlit Application

```bash
streamlit run app.py
```

If Streamlit is not added to your system PATH, use:

```bash
python -m streamlit run app.py
```

---

## Step 3: Open the Application

After the server starts, open your browser and visit:

```
http://localhost:####
```

You can now enter soil and environmental parameters to receive crop recommendations.

## 📈 Workflow

1. User enters soil nutrient values.
2. Environmental parameters are provided.
3. Data is preprocessed.
4. The stacking ensemble model predicts crop probabilities.
5. The crop with the highest probability is recommended.
6. SHAP explains the prediction by showing feature importance.

---

## 🔍 Explainable AI

The application integrates **SHAP (SHapley Additive Explanations)** to improve transparency by displaying:

- Feature importance
- SHAP summary plots
- Feature contribution analysis
- Model interpretation

---

## 🎯 Applications

- Smart Farming
- Precision Agriculture
- Agricultural Decision Support
- Educational Projects
- Machine Learning Research

---

## 🚀 Future Enhancements

- Weather API Integration
- Fertilizer Recommendation
- Crop Yield Prediction
- Disease Detection
- IoT Sensor Integration
- Mobile Application
- Multi-language Support

---

## 👨‍💻 Project Team

This project was developed as a collaborative academic project by:

- **Abhinav Manoj**
- **Gagan Dev P V**
- **Jyothis C R**
- **Adithyan M C**

### Institution

**St. Thomas College of Engineering and Technology, Mattanur**

### Department

**Computer Science & Engineering (Data Science)**

---

## 🙏 Acknowledgements

We sincerely thank:

- St. Thomas College of Engineering and Technology, Mattanur
- Department of Computer Science & Engineering (Data Science)
- Faculty mentors for their guidance
- The open-source communities behind Streamlit, Scikit-Learn, XGBoost, and SHAP

---

## 📜 License

This project is intended for educational and research purposes.

Feel free to use, modify, and distribute it with proper attribution.

---

## ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork this repository

📢 Share it with others

Happy Farming! 🌱
