import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import os

# === Configurations ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128
os.makedirs("./Best_Model", exist_ok=True)
save_path = "./Best_Model/best_svr_pIC50"

# === 데이터 불러오기 ===
df = pd.read_csv("./Raw_Data/train_df_final_min.csv")
df = df.dropna(subset=["Canonical_Smiles", "pIC50"])
df["pIC50"] = df["pIC50"].astype(float)

# === Preprocessing: pIC50 값 필터링 (10e-6 nM 이하 제거) ===
pIC50_cutoff = -np.log10(10e-6)
df = df[(df["pIC50"] >= pIC50_cutoff)]

# === 사전 학습 모델 및 토크나이저 불러오기 ===
tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
chemberta = AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM').to(device)

# === Dataset Class for Feature Extraction ===
class ChemDataset(Dataset):
    def __init__(self, smiles_list, labels):
        self.smiles_list = smiles_list
        self.labels = labels

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        inputs = tokenizer(smi, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs, label

# === 특성 추출 함수 ===
def extract_chemberta_features(model, dataloader, device):
    """
    Extract ChemBERTa features from SMILES strings using mean pooling
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            inputs, batch_labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 토크나이저 통과 후, ChemBERTa 모델 통과
            outputs = model(**inputs)

            # CLS 토큰 대신, mean pooling 사용 
            attention_mask = inputs['attention_mask']
            # Attention mask 적용 후, mean pooling 진행
            masked_outputs = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            pooled = masked_outputs.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            features.append(pooled.cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    return np.vstack(features), np.array(labels)

# === 학습 데이터 ===
train_df = df.copy()

print(f"Training set size: {len(train_df)}")

# === 데이터셋 생성 및 데이터로더 생성 ===
train_dataset = ChemDataset(train_df["Canonical_Smiles"].tolist(), train_df["pIC50"].tolist())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# === ChemBERTa 특성 추출 ===
print("Extracting ChemBERTa features from training set...")
X_train, y_train = extract_chemberta_features(chemberta, train_loader, device)

print(f"Feature matrix shape - Train: {X_train.shape}")

# === 보간 데이터 추가 ===
print("\n=== Adding Interpolated Data ===")

# pIC50 9초과 10이하와 13인 데이터 찾기 (보간 데이터 생성)
mask_9_10 = (train_df["pIC50"] > 9) & (train_df["pIC50"] <= 10)
mask_13 = (train_df["pIC50"] == 13)

print(f"pIC50 9-10 데이터 개수: {mask_9_10.sum()}")
print(f"pIC50 13 데이터 개수: {mask_13.sum()}")

if mask_9_10.sum() > 0 and mask_13.sum() > 0:
    # 9-10 구간의 각 샘플과 13 구간의 각 샘플을 조합
    embeddings_9_10 = X_train[mask_9_10]
    pIC50_9_10 = train_df.loc[mask_9_10, "pIC50"].values
    
    embeddings_13 = X_train[mask_13]
    pIC50_13 = train_df.loc[mask_13, "pIC50"].values
    
    # 모든 조합으로 interpolated 데이터 생성
    interpolated_embeddings = []
    interpolated_pIC50s = []
    
    for i in range(len(embeddings_9_10)):
        for j in range(len(embeddings_13)):
            # 각 9-10 샘플과 각 13 샘플의 평균
            interpolated_embedding = (embeddings_9_10[i] + embeddings_13[j]) / 2
            interpolated_pIC50 = (pIC50_9_10[i] + pIC50_13[j]) / 2
            
            interpolated_embeddings.append(interpolated_embedding)
            interpolated_pIC50s.append(interpolated_pIC50)
    
    # interpolated 데이터를 training set에 추가
    X_train = np.vstack([X_train, np.array(interpolated_embeddings)])
    y_train = np.concatenate([y_train, interpolated_pIC50s])
    
    print(f"Generated {len(embeddings_9_10)} × {len(embeddings_13)} = {len(interpolated_embeddings)} interpolated samples")
    print(f"Final feature matrix shape: {X_train.shape}")
    print(f"Final training set size: {len(y_train)}")
else:
    print("Not enough data for interpolation")

# === SVR 모델 학습 ===
print("\n=== Training SVR Model (default hyperparameters) ===")
svr = SVR(C=0.8, epsilon=0.005)  # default: kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'
svr.fit(X_train, y_train)

# === 예측 결과 계산 ===
print("Making predictions...")
train_preds = svr.predict(X_train)

# === 성능 지표 계산 ===
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
train_mae = mean_absolute_error(y_train, train_preds)
train_r2 = r2_score(y_train, train_preds)

print(f"\n🎯 Model Performance:")
print(f"Training - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")

# === 모델 저장 ===
print(f"Saving model to {save_path}...")
joblib.dump(svr, save_path)