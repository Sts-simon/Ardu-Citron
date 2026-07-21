import os
import glob
import json
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Configuration
DATA_DIR = "./cnn_roi_dataset"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

# ==============================================================================
# 1. DATASET PYTORCH AVEC NORMALISATION COMPLÈTE DES LABELS
# ==============================================================================
class DronePoseDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        if len(self.image_paths) == 0:
            raise RuntimeWarning(f"Aucune image trouvée dans {data_dir}. Lance d'abord generate_dataset_v2.py !")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        json_path = img_path.replace(".png", ".json")
        
        # Charger l'image (128x128x3) -> RVB -> Normalisation [0, 1] -> Format CHW
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Charger les labels bruts
        with open(json_path, 'r') as f:
            data = json.load(f)
        labels = data["target_pose_xyz_rpy"] # [X, Y, Z, Roll, Pitch, Yaw]
        
        # ✨ NORMALISATION MATHÉMATIQUE BORNÉE
        # Positions X et Y : ramenées par rapport à une dérive générale de [-3.0, 3.0] mètres
        x_norm = labels[0] / 3.0
        y_norm = labels[1] / 3.0
        # Altitude Z : configurée strictement entre 2.0m et 6.0m -> ramenée entre [0, 1]
        z_norm = (labels[2] - 2.0) / (6.0 - 2.0)
        
        # Angles : Normalisation sur les plages dynamiques max du simulateur (35° Roll, 20° Pitch, 45° Yaw)
        roll_norm = labels[3] / 35.0   # Évolue entre -1 et 1
        pitch_norm = labels[4] / 20.0  # Évolue entre -1 et 1
        yaw_norm = labels[5] / 45.0    # Évolue entre -1 et 1
        
        normalized_labels = [x_norm, y_norm, z_norm, roll_norm, pitch_norm, yaw_norm]
        
        return img_tensor, torch.tensor(normalized_labels, dtype=torch.float32)

# ==============================================================================
# 2. ARCHITECTURE DU CNN (TINY DRONE LOCALIZER)
# ==============================================================================
class TinyDroneLocalizer(nn.Module):
    def __init__(self):
        super(TinyDroneLocalizer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 128x128 -> 64x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64x64 -> 32x32
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32 -> 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16 -> 8x8
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 6) # Sortie : 6 neurones pour nos 6 états normalisés
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# ==============================================================================
# 3. BOUCLE D'ENTRAÎNEMENT PONDÉRÉE ET EXPORT CHEKPOINTS ONNX
# ==============================================================================
def train():
    print("🚀 Préparation de l'entraînement...")
    dataset = DronePoseDataset(DATA_DIR)
    
    # Split 80% train / 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = TinyDroneLocalizer()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Poids des tâches pour forcer l'apprentissage sur les angles récalcitrants
    w_pos = 1.0
    w_angle = 5.0
    
    os.makedirs("checkpoints", exist_ok=True)
    print(f"📈 Début de l'apprentissage ({EPOCHS} époques)...")
    
    # Tenseur d'entrée factice réutilisé pour l'export ONNX à chaque époque
    dummy_input = torch.randn(1, 3, 128, 128)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            
            # Décomposition des erreurs pour équilibrer le gradient
            loss_x = nn.functional.mse_loss(outputs[:, 0], targets[:, 0])
            loss_y = nn.functional.mse_loss(outputs[:, 1], targets[:, 1])
            loss_z = nn.functional.mse_loss(outputs[:, 2], targets[:, 2])
            
            loss_r = nn.functional.mse_loss(outputs[:, 3], targets[:, 3])
            loss_p = nn.functional.mse_loss(outputs[:, 4], targets[:, 4])
            loss_yaw = nn.functional.mse_loss(outputs[:, 5], targets[:, 5])
            
            # Somme pondérée de la Loss
            loss = w_pos * (loss_x + loss_y + loss_z) + w_angle * (loss_r + loss_p + loss_yaw)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation rapide
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                outputs = model(images)
                
                loss_x = nn.functional.mse_loss(outputs[:, 0], targets[:, 0])
                loss_y = nn.functional.mse_loss(outputs[:, 1], targets[:, 1])
                loss_z = nn.functional.mse_loss(outputs[:, 2], targets[:, 2])
                loss_r = nn.functional.mse_loss(outputs[:, 3], targets[:, 3])
                loss_p = nn.functional.mse_loss(outputs[:, 4], targets[:, 4])
                loss_yaw = nn.functional.mse_loss(outputs[:, 5], targets[:, 5])
                
                loss = w_pos * (loss_x + loss_y + loss_z) + w_angle * (loss_r + loss_p + loss_yaw)
                val_loss += loss.item() * images.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] -> Loss Train: {epoch_loss:.5f} | Loss Val: {epoch_val_loss:.5f}")
        
        # 💾 1. Sauvegarde du checkpoint complet au format PyTorch (.pth)
        checkpoint_path = f"checkpoints/tiny_drone_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': epoch_val_loss,
        }, checkpoint_path)
        
        # 📦 2. ✨ NOUVEAUTÉ : Export SIMULTANÉ du checkpoint au format ONNX
        onnx_checkpoint_path = f"checkpoints/tiny_drone_epoch_{epoch+1}.onnx"
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_checkpoint_path, 
            export_params=True, 
            opset_version=11, 
            do_constant_folding=True, 
            input_names=['input_roi'], 
            output_names=['output_pose']
        )
        print(f"   ↳ 💾 Checkpoints sauvegardés : {checkpoint_path} ET {onnx_checkpoint_path}")
        
    # Sauvegarde finale classique des poids
    torch.save(model.state_dict(), "tiny_drone_cnn.pth")
    print("\n✅ Entraînement fini ! Les poids PyTorch finaux sont dans 'tiny_drone_cnn.pth'.")

    # Exportation finale du modèle ONNX principal
    print("📦 Génération du modèle ONNX de production final...")
    onnx_final_path = "tiny_drone_localizer.onnx"
    model.eval()
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_final_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True, 
        input_names=['input_roi'], 
        output_names=['output_pose']
    )
    print(f"✨ Export final terminé avec succès ! Modèle prêt : '{onnx_final_path}'")

if __name__ == "__main__":
    train()
