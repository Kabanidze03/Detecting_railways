import torch
import catboost
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnext101_32x8d
from torch.optim import AdamW, lr_scheduler
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split


# model = nn.Sequential(*list(model.children()))


def image_transform(img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transform(img).unsqueeze(0)


def download_model() -> nn.Module:
    model = resnext101_32x8d(weights='ResNeXt101_32X8D_Weights.IMAGENET1K_V2')
    model = nn.Sequential(*list(model.children())[:-1])
    
    return model

def download_classifier() -> catboost.core.CatBoostClassifier:
    classifier = CatBoostClassifier()
    classifier.load_model("classifier.cbm")

    return classifier


def feature_extraction(img_tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(img_tensor)

    return out

def post_processing_image(img_tensor: torch.Tensor) -> np.ndarray:
    np_img = img_tensor.view(img_tensor.shape[0], -1).squeeze(0).cpu().numpy()

    return np_img

def make_probability_diagram(prediction: np.ndarray):

    plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle")
    # Подготовим данные
    probabilities = [prediction[0], prediction[1]]
    labels = ['Нет ж/д', 'Есть ж/д']
    df = pd.DataFrame({'Класс': labels, 'Вероятность': probabilities})

    # Создаём фигуру и ось
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='Класс', y='Вероятность', ax=ax, palette="Blues")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Вероятность")
    ax.set_title("Результат классификации")

    # Добавим подписи над столбцами
    for i, p in enumerate(probabilities):
        ax.text(i, p + 0.02, f"{p:.2f}", ha='center', fontweight='bold')

    st.pyplot(fig)

