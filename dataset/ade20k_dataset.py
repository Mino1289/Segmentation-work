import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors
import numpy as np
from datasets import load_dataset


class ADE20KDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Extraction de l'image et du premier masque sémantique
        image = item["image"].convert("RGB")
        mask_img = item["annotation"]

        # Le dataset de test ne fournit pas les annotations (None). On crée un masque vide par défaut.
        if mask_img is None:
            w, h = image.size
            mask_decoded = np.zeros((h, w), dtype=np.int32)
        else:
            # Convertir le masque (palette 2D) en format tensor
            mask_decoded = np.array(mask_img, dtype=np.int32)

        # Convertir l'image et le masque au format Tensor natif de torchvision.v2
        # Le masque est converti en Tensor de type long avec une dimension de channel [1, H, W]
        image = tv_tensors.Image(v2.functional.to_image(image))
        mask = tv_tensors.Mask(
            torch.as_tensor(mask_decoded, dtype=torch.long).unsqueeze(0)
        )

        if self.transform is not None:
            # Grâce à torchvision.v2, la transformation est appliquée simultanément à l'image et au masque.
            # Le redimensionnement (Resize) applique automatiquement l'interpolation NEAREST au masque
            # et BILINEAR à l'image, préservant ainsi l'intégrité des IDs de classes.
            image, mask = self.transform(image, mask)

        # On supprime la dimension du channel du masque pour obtenir un shape [H, W] (requis par CrossEntropyLoss)
        return image, mask.squeeze(0)


if __name__ == "__main__":
    print("Chargement du dataset depuis Hugging Face...")
    raw_dataset = load_dataset("merve/scene_parse_150")

    print("Dataset divisé avec succès :")
    print(f" - Train : {len(raw_dataset['train'])} images")
    print(f" - Validation : {len(raw_dataset['validation'])} images")
    print(f" - Test : {len(raw_dataset['test'])} images\n")
    
    # Pipeline Train : Augmentation de données pour maximiser les performances du modèle
    train_transforms = v2.Compose([
        v2.Resize((512, 512)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Modifie uniquement l'image, ignore le masque
        v2.ToDtype(torch.float32, scale=True),                         # Normalise l'image [0, 1], ignore le masque
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalise uniquement l'image
    ])

    # Pipeline Validation & Test : Pas d'augmentation de données, uniquement la mise au format
    val_test_transforms = v2.Compose([
        v2.Resize((512, 512)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ADE20KDataset(raw_dataset["train"], transform=train_transforms)
    val_dataset   = ADE20KDataset(raw_dataset["validation"], transform=val_test_transforms)
    test_dataset  = ADE20KDataset(raw_dataset["test"], transform=val_test_transforms)

    # Paramètres de performance pour l'entraînement sur GPU
    BATCH_SIZE = 16
    NUM_WORKERS = 8       # Multi-processing CPU pour charger les images en parallèle
    PIN_MEMORY = True     # Accélère considérablement le transfert CPU -> GPU (VRAM)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Pour le test/benchmark, un batch size de 1 est idéal pour évaluer la mIoU image par image de manière précise
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=PIN_MEMORY)


    for images, masks in train_loader:
        print(f" -> Images (Train) shape : {images.shape} (Attendu: [{BATCH_SIZE}, 3, 512, 512])")
        print(f" -> Masks  (Train) shape : {masks.shape} (Attendu: [{BATCH_SIZE}, 512, 512])")
        print(f" -> Masks  Dtype        : {masks.dtype} (Attendu: torch.int64 / long)")
        break

    print("\nVérification d'un batch du Test Loader...")
    for images, masks in test_loader:
        print(f" -> Images (Test) shape  : {images.shape} (Attendu: [1, 3, 512, 512])")
        print(f" -> Masks  (Test) shape  : {masks.shape} (Attendu: [1, 512, 512])")
        break