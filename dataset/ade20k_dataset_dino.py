import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors
import numpy as np
from datasets import load_dataset


class ADE20KDatasetDino(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item["image"].convert("RGB")
        mask_img = item["annotation"]

        if mask_img is None:
            w, h = image.size
            # Utilisation de 255 comme standard PyTorch pour ignore_index
            mask_decoded = np.full((h, w), 255, dtype=np.int32)
        else:
            mask_decoded = np.array(mask_img, dtype=np.int32)
            mask_decoded = mask_decoded - 1
            # Les pixels à -1 (anciennement 0/fond) passent à 255 pour être ignorés par la Loss
            mask_decoded[mask_decoded == -1] = 255

        image = tv_tensors.Image(v2.functional.to_image(image))
        mask = tv_tensors.Mask(
            torch.as_tensor(mask_decoded, dtype=torch.long).unsqueeze(0)
        )

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask.squeeze(0)


def get_dataloaders(batch_size=4, num_workers=0, resolution=518):
    print("Chargement du dataset depuis Hugging Face...")
    raw_dataset = load_dataset("merve/scene_parse_150")

    SCALE_RANGE = (0.5, 2.0)
    train_transforms = v2.Compose(
        [
            v2.RandomResize(
                min_size=int(resolution * SCALE_RANGE[0]),
                max_size=int(resolution * SCALE_RANGE[1]),
                antialias=True,
            ),
            v2.RandomCrop(
                size=(resolution, resolution),
                pad_if_needed=True,
                fill={tv_tensors.Image: 0, tv_tensors.Mask: 255},
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_test_transforms = v2.Compose(
        [
            v2.Resize(size=resolution, antialias=True),
            v2.CenterCrop(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ADE20KDatasetDino(raw_dataset["train"], transform=train_transforms)
    val_dataset = ADE20KDatasetDino(
        raw_dataset["validation"], transform=val_test_transforms
    )

    PIN_MEMORY = torch.backends.mps.is_available() or torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader


# A SUPPRIMER
if __name__ == "__main__":
    print("Chargement du dataset depuis Hugging Face...")
    raw_dataset = load_dataset("merve/scene_parse_150")

    print("Dataset divisé avec succès :")
    print(f" - Train : {len(raw_dataset['train'])} images")
    print(f" - Validation : {len(raw_dataset['validation'])} images")
    print(f" - Test : {len(raw_dataset['test'])} images\n")

    # Résolution stricte pour DINOv2 (multiple de 14)
    RESOLUTION = 518

    # Pipeline Train
    SCALE_RANGE = (0.5, 2.0)
    train_transforms = v2.Compose(
        [
            v2.RandomResize(
                min_size=int(RESOLUTION * SCALE_RANGE[0]),
                max_size=int(RESOLUTION * SCALE_RANGE[1]),
                antialias=True,
            ),
            v2.RandomCrop(
                size=(RESOLUTION, RESOLUTION),
                pad_if_needed=True,
                fill={tv_tensors.Image: 0, tv_tensors.Mask: 255},
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Pipeline Validation / Test
    val_test_transforms = v2.Compose(
        [
            v2.Resize(size=RESOLUTION, antialias=True),
            v2.CenterCrop(size=(RESOLUTION, RESOLUTION)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ADE20KDatasetDino(raw_dataset["train"], transform=train_transforms)
    val_dataset = ADE20KDatasetDino(
        raw_dataset["validation"], transform=val_test_transforms
    )
    test_dataset = ADE20KDatasetDino(raw_dataset["test"], transform=val_test_transforms)

    # Paramètres optimisés pour l'architecture matérielle
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    PIN_MEMORY = torch.backends.mps.is_available() or torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Vérification
    for images, masks in train_loader:
        print(
            f" -> Images (Train) shape : {images.shape} (Attendu: [{BATCH_SIZE}, 3, {RESOLUTION}, {RESOLUTION}])"
        )
        print(
            f" -> Masks  (Train) shape : {masks.shape} (Attendu: [{BATCH_SIZE}, {RESOLUTION}, {RESOLUTION}])"
        )
        print(f" -> Masks  Dtype         : {masks.dtype} (Attendu: torch.int64)")
        break

    print("\nVérification d'un batch du Test Loader...")
    for images, masks in test_loader:
        print(
            f" -> Images (Test) shape  : {images.shape} (Attendu: [1, 3, {RESOLUTION}, {RESOLUTION}])"
        )
        print(
            f" -> Masks  (Test) shape  : {masks.shape} (Attendu: [1, {RESOLUTION}, {RESOLUTION}])"
        )
        break
