import os
import glob
import json
import cv2
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

TRAIN_DIR = "./Dataset_CNN/train"
VAL_DIR = "./Dataset_CNN/val"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0005  # LR légèrement plus doux
NUM_WORKERS = 4

# ==============================================================================
# 1. DATASET : CONVERSION DU YAW EN SIN / COS
# ==============================================================================
class DronePoseDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        if len(self.image_paths) == 0:
            raise RuntimeWarning(f"Aucune image trouvée dans {data_dir} !")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        json_path = img_path.replace(".png", ".json")
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        labels = data["target_pose_xyz_rpy"] # [X, Y, Z, Roll, Pitch, Yaw]
        
        # Normalisation des positions
        x_norm = labels[0] / 6.0
        y_norm = labels[1] / 6.0
        z_norm = (labels[2] - 2.0) / (6.0 - 2.0)
        
        # Angles Roll / Pitch
        roll_norm = labels[3] / 35.0
        pitch_norm = labels[4] / 20.0
        
        # ✨ FIX CRUCIAL : Yaw représenté par sin(rad) et cos(rad)
        yaw_rad = math.radians(labels[5])
        sin_yaw = math.sin(yaw_rad)
        cos_yaw = math.cos(yaw_rad)
        
        # 7 cibles au lieu de 6 !
        normalized_labels = [x_norm, y_norm, z_norm, roll_norm, pitch_norm, sin_yaw, cos_yaw]
        
        return img_tensor, torch.tensor(normalized_labels, dtype=torch.float32)

# ==============================================================================
# 2. ARCHITECTURE RENFORCÉE AVEC BATCHNORMALIZATION
# ==============================================================================
class TinyDroneLocalizer(nn.Module):
    def __init__(self):
        super(TinyDroneLocalizer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 128 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 7) # 7 sorties
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# ==============================================================================
# 3. ENTRAÎNEMENT
# ==============================================================================
def train():
    print("🚀 Préparation de l'entraînement V2...")
    train_dataset = DronePoseDataset(TRAIN_DIR)
    val_dataset = DronePoseDataset(VAL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDroneLocalizer().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss() # Plus stable que MSE pour la régression
    
    print(f"📈 Début de l'apprentissage ({EPOCHS} époques)...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] -> Loss Train: {epoch_loss:.5f} | Loss Val: {epoch_val_loss:.5f}")

    # Export ONNX
    print("📦 Génération du modèle ONNX...")
    model.eval().cpu()
    dummy_input = torch.randn(1, 3, 128, 128)
    torch.onnx.export(
        model, dummy_input, "tiny_drone_localizer.onnx",
        export_params=True, opset_version=18,
        input_names=['input_roi'], output_names=['output_pose']
    )
    print("✨ Modèle ONNX généré !")

if __name__ == "__main__":
    train()
