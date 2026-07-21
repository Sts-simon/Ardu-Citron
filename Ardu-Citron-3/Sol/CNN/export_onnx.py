import os
import torch
from train_cnn import TinyDroneLocalizer

def export():
    # 1. Initialiser le modèle
    model = TinyDroneLocalizer()
    
    # Déterminer le bon chemin du modèle
    model_path = "tiny_drone_epoch_14.pth"
    if not os.path.exists(model_path):
        model_path = "checkpoints/tiny_drone_epoch_3.pth" # Force l'époque 3 si le final n'existe pas
        
    if not os.path.exists(model_path):
        print(f"❌ Impossible de trouver le fichier de poids.")
        return

    # 2. Charger les poids intelligemment (gestion du dictionnaire)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print(f"💾 Extraction des poids depuis le checkpoint complet : {model_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"📦 Chargement direct des poids bruts : {model_path}")
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # 3. Créer une fausse image (Dummy Input) au format attendu (Batch, Channels, Height, Width)
    # Notre réseau attend une image de taille (1, 3, 128, 128)
    dummy_input = torch.randn(1, 3, 128, 128)
    
    # 4. Exporter vers ONNX
    onnx_path = "tiny_drone_localizer.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,        # Stocke les poids entraînés dans le fichier
        opset_version=12,          # Version standard et stable d'ONNX
        do_constant_folding=True,  # Optimise les opérations constantes à l'export
        input_names=['input_roi'], # Nom de la couche d'entrée (utile pour le code Rust)
        output_names=['output_pose'] # Nom de la couche de sortie
    )
    
    print(f"🚀 Modèle exporté avec succès sous '{onnx_path}' !")

if __name__ == "__main__":
    export()
