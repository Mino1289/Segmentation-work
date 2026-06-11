import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.dinov2 import DinoV2, DinoV2ADE20K, DinoV2Config
from dataset.ade20k_dataset_dino import get_dataloaders

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
    config = DinoV2Config(name="S")
    backbone = DinoV2(config=config, image_size=518, patch_size=14)
    
    model = DinoV2ADE20K(backbone=backbone, num_classes=150).to(device)
    
    print("Préparation des données...")
    train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0, resolution=518)
    
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    #ignorer les pixels de padding/fond à 255 de ADE20K
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
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