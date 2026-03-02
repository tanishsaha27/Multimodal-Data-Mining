# ---------------------------------------
# MULTI-MODAL DATA MINING PROJECT
# Name: Tanish Saha
# ---------------------------------------

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------
data = pd.read_csv("dataset.csv")

# ---------------------------------------
# TEXT FEATURE EXTRACTION (TF-IDF)
# ---------------------------------------
tfidf = TfidfVectorizer(max_features=100)
text_features = tfidf.fit_transform(data['caption']).toarray()

# ---------------------------------------
# IMAGE FEATURE EXTRACTION (ResNet18)
# ---------------------------------------
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_image_features(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = resnet(img)
        return features.numpy().flatten()
    except:
        return np.zeros(512)  # fallback if error

image_features = np.array([
    extract_image_features(path) for path in data['image_path']
])

# ---------------------------------------
# FEATURE FUSION
# ---------------------------------------
combined_features = np.hstack((text_features, image_features))

# ---------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    combined_features,
    data['label'],
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# MODEL TRAINING
# ---------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------------------
# PREDICTION
# ---------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------
# EVALUATION
# ---------------------------------------
print("\n===== RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))