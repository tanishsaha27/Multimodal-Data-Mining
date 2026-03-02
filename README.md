# Multi-Modal Data Mining (Text + Image)

## 📌 Overview
This project demonstrates multi-modal data mining by combining **text (captions)** and **image data** for classification. It shows how integrating multiple data types improves prediction accuracy compared to single-modal approaches.

---

## 🎯 Objective
To build a system that:
- Extracts features from text and images
- Combines them using feature fusion
- Performs classification using machine learning

---

## ⚙️ Methodology
- Text → TF-IDF feature extraction  
- Image → ResNet18 (CNN) feature extraction  
- Fusion → Concatenation of features  
- Model → Logistic Regression  
- Evaluation → Accuracy, Precision, Recall, F1-score  

---

## 📂 Project Structure
multimodal-data-mining/
│
├── multimodal_model.py
├── dataset.csv
├── images/
└── README.md


---

## 🚀 How to Run
Install dependencies: pip install numpy pandas scikit-learn torch torchvision pillow

Run the code: python multimodal_model.py


---

## 📊 Result
Multi-modal model performs better than single-modal models by capturing both semantic (text) and visual (image) information.

---

## 👤 Author
**Tanish Saha**  
B.Tech CSE (6th Sem)  
Kalyani Government Engineering College
