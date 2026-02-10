import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import joblib
from rdkit import Chem
from rdkit.Chem import MolStandardize
import os

# === Configurations ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128
model_path = "./Best_Model/best_svr_pIC50"
test_data_path = "./Raw_Data/test.csv"
output_path = "./Final_Submission/testset_predictions.csv"

os.makedirs("./Final_Submission", exist_ok=True)

# === Canonicalization function ===
def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        clean = MolStandardize.rdMolStandardize.Cleanup(mol)
        return Chem.MolToSmiles(clean, canonical=True)
    except:
        return None

# === 테스트 데이터 불러오기 ===
print("Loading test data...")
test_df = pd.read_csv(test_data_path)
print(f"Original test data size: {len(test_df)}")

test_df["Canonical_Smiles"] = test_df["Smiles"].apply(canonicalize)
test_df = test_df.dropna(subset=["Canonical_Smiles"])
print(f"After canonicalization: {len(test_df)} samples")

# === 토크나이저 및 ChemBERTa 모델 불러오기 ===
print("Loading tokenizer and ChemBERTa model...")
tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')
chemberta = AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM').to(device)

# === 데이터셋 클래스 정의 ===
class InferenceDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        inputs = tokenizer(smi, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs

# === ChemBERTa mean pooling 특성 추출 ===
def extract_chemberta_features(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            masked_outputs = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            pooled = masked_outputs.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            features.append(pooled.cpu().numpy())
    return np.vstack(features)

# === 테스트 데이터셋 준비 ===
test_dataset = InferenceDataset(test_df["Canonical_Smiles"].tolist())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# === ChemBERTa 특성 추출 ===
print("Extracting ChemBERTa features from test set...")
X_test = extract_chemberta_features(chemberta, test_loader, device)
print(f"Test feature matrix shape: {X_test.shape}")

# === 학습된 SVR 모델 불러오기 ===
print("Loading SVR model...")
svr_model = joblib.load(model_path)

# === 예측 결과 계산 ===
print("Predicting with SVR...")
predictions_pIC50 = svr_model.predict(X_test)

# === pIC50를 IC50 nM로 변환 ===
predictions_nM = 10 ** (9 - predictions_pIC50)

# === 제출 형식 결과 생성 ===
submission_df = pd.DataFrame({
    'ID': test_df["ID"].values,
    'ASK1_IC50_nM': predictions_nM
})

# === 결과 저장 ===
submission_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
