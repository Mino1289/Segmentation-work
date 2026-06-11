import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.dinov2 import DinoV2, DinoV2ADE20K, DinoV2Config
from dataset.ade20k_dataset_dino import get_dataloaders

import torch
import torch.nn as nn
import torch.nn.functional as F


##FONCTION VIBECODEE
class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -1, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, true_masks: torch.Tensor) -> torch.Tensor:
        # 1. Calcul de la Cross Entropy standard
        ce_loss = self.ce(logits, true_masks)
        
        # 2. Préparation pour la Dice Loss
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1) # [B, C, H, W]
        
        # Isoler les pixels valides (ignorer les -1)
        valid_pixels = true_masks != self.ignore_index
        
        # Créer une copie sécurisée pour le One-Hot (remplacer les -1 par 0 temporairement)
        safe_masks = true_masks.clone()
        safe_masks[~valid_pixels] = 0
        
        # One-hot encoding des masques : [B, C, H, W]
        true_1hot = F.one_hot(safe_masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Appliquer le filtre de validité pour que le fond ne compte pas dans la Dice Loss
        valid_pixels = valid_pixels.unsqueeze(1).float()
        probs = probs * valid_pixels
        true_1hot = true_1hot * valid_pixels
        
        # 3. Calcul de la Dice Loss
        # Somme sur le batch et les dimensions spatiales
        dims = (0, 2, 3) 
        intersection = torch.sum(probs * true_1hot, dims)
        cardinality = torch.sum(probs + true_1hot, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        # 4. Score combiné
        return ce_loss + dice_loss

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    #barre de chargement
    pbar = tqdm(dataloader, desc='Entrainement')
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        return running_loss / len(dataloader)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.backends.cuda.is_available() else "cpu")
    print(f"Périphérique utilisé : {device}")
    
    print("Initialisation du modèle...")
    config = DinoV2Config(name="B")
    backbone = DinoV2(config=config, image_size=518, patch_size=14)
    #Si taille = B, alors embed_dim = 768
    model = DinoV2ADE20K(backbone=backbone, num_classes=150, embed_dim=768).to(device)
    
    print("Préparation des données...")
    train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0, resolution=518)
    
    EPOCHS = 10
    LEARNING_RATE = 5e-3
    WEIGHT_DECAY = 1e-4
    
    #ignorer les pixels de padding/fond à 255 de ADE20K
    criterion = DiceCrossEntropyLoss(ignore_index=-1)
    
    #passer uniquement les paramètres non gelés à l'optimiseur
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    #???
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print("Début entrainement... \n")
    best_val_loss = float("inf")
    
    for epoch in range(EPOCHS):
        print(f"--- Époque {epoch + 1}/{EPOCHS} ---")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Résumé : Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_dinov2_ade20k.pth")
            print("Nouveau meilleur modèle sauvegardé!")
        print("\n")