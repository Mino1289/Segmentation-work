import torch
from torch import nn
from typing import Tuple

#from visualization.dino_visualization import visualize_attention
from models.layers.dino_block import DinoBlock
from models.layers.patch_embedding import PatchEmbedding
from models.layers.dino_segmentation_head import DinoSegmentationHead


class DinoV2Config:
    def __init__(self, name:str = 'S'):
        ## tirées de la table 17 de l'article de DinoV2
        configs = {
            'S': {'embed_dim':384, 'num_heads':6, 'depth':12},
            'B': {'embed_dim':768, 'num_heads':12, 'depth':18},
            'L': {'embed_dim':1024, 'num_heads':16, 'depth':24}
        }
        if name not in configs:
            raise ValueError(
                f"Taille '{name}' non reconnue. Choisissez parmi 'S', 'B', 'L'."
            )
        self.name = name
        self.embed_dim = configs[name]['embed_dim']
        self.num_heads = configs[name]['num_heads']
        self.depth = configs[name]['depth']

class DinoV2(nn.Module):
    def __init__(
        self,
        config: DinoV2Config = DinoV2Config(name='S'),
        image_size: int = 518,
        patch_size: int = 14
    ):
        
        super().__init__()
        
        self.config = config
        
        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.depth = self.config.depth
        
        num_patches = (image_size // patch_size)**2
        
        self.patch_embed = PatchEmbedding(patch_size=patch_size, emb_size=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, self.embed_dim))
        self.blocks = nn.ModuleList([DinoBlock(self.embed_dim, self.num_heads) for _ in range(self.depth)])
        self.norm = nn.LayerNorm(normalized_shape=self.embed_dim, eps=1e-6)
        
    def forward(self, x):
        x = self.patch_embed(x)
        #sortie [B, H, W, C]
        
        #ici on veut [B, N, C]
        b, h, w, c = x.shape
        x = x.reshape(b, -1, c)
        
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token,x), dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x= self.norm(x)
        
        #supprimer le CLS token
        x = x[:, 1:, :]
        
        x = x.reshape(b, h, w,c)
        x = x.permute(0,3,1,2).contiguous()
        
        return x
        
    #FONCTION VIBECODEE
    def load_from_timm(self, timm_model: nn.Module):
        state_dict_timm = timm_model.state_dict()
        state_dict_custom = self.state_dict()

        new_state_dict = {}
        loaded_keys = []
        ignored_keys = []
        mismatch_keys = []

        for k_timm, v_timm in state_dict_timm.items():
            # 1. Ignorer la tête de classification (head)
            # timm ajoute toujours un classifieur linéaire à la fin, inutile en segmentation
            if k_timm.startswith("head."):
                ignored_keys.append(k_timm)
                continue
                
            k_custom = k_timm

            # 2. Mapping direct : on vérifie si la clé existe dans ton modèle
            if k_custom in state_dict_custom:
                v_custom = state_dict_custom[k_custom]
                
                # 3. Vérification de sécurité sur les dimensions
                if v_custom.shape == v_timm.shape:
                    new_state_dict[k_custom] = v_timm
                    loaded_keys.append(k_custom)
                else:
                    # Arrive souvent si on charge des poids entraînés sur du 224x224 
                    # et qu'on a instancié notre modèle pour du 512x512 (pos_embed mismatch)
                    mismatch_keys.append(f"{k_custom}: custom {v_custom.shape} != timm {v_timm.shape}")
            else:
                ignored_keys.append(k_timm)

        # 4. Chargement effectif des poids
        missing_in_custom, unexpected_in_custom = self.load_state_dict(new_state_dict, strict=False)

        # 5. Logs pour le débogage de l'architecture
        print(f" {len(loaded_keys)} tenseurs de poids chargés avec succès.")
        
        if mismatch_keys:
            print(f" {len(mismatch_keys)} erreurs de dimensions (ignorées) :")
            for m in mismatch_keys[:5]: print(f"  - {m}")
                
        if missing_in_custom:
            print(f" Clés manquantes dans ton modèle (non chargées) : {len(missing_in_custom)}")
            for m in missing_in_custom[:5]: print(f"  - {m}")
        return self
    
    
class DinoV2ADE20K(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes :int=150, embed_dim:int=384):
        super().__init__()
        self.backbone = backbone
        
        #freeze les poids du backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.head = DinoSegmentationHead(in_channels=embed_dim, num_classes = num_classes)
    
    def forward(self, x):
        target_size = (x.shape[2], x.shape[3])
        
        with torch.no_grad():
            features = self.backbone(x)
        masks = self.head(features, target_size)
        return masks
        
    
#FONCTION VIBECODEE
if __name__ == "__main__":
    import timm
    import torch
    from torch import nn

    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    size = "S"  # "S", "B", ou "L"

    # Mapping strict vers les poids officiels DINOv2 dans timm
    timm_models = {
        "S": "vit_small_patch14_dinov2.lvd142m",
        "B": "vit_base_patch14_dinov2.lvd142m",
        "L": "vit_large_patch14_dinov2.lvd142m"
    }

    # 2. Instanciation du modèle timm
    # num_classes=0 désactive automatiquement la création de la tête linéaire (head)
    timm_model = timm.create_model(timm_models[size], pretrained=True, num_classes=0)
    timm_model = timm_model.eval().to(device)

    # 3. Instanciation de ton modèle
    config = DinoV2Config(name=size)
    mymodel = DinoV2(config=config, image_size=518, patch_size=14)
    mymodel.load_from_timm(timm_model)
    mymodel = mymodel.eval().to(device)

    # 4. Inférence
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 518, 518).to(device)
        
        # Ta sortie : [B, C, H_patch, W_patch]
        output = mymodel(dummy_input)

        # Sortie timm : forward_features renvoie la séquence brute [B, N+1, C]
        output_timm_raw = timm_model.forward_features(dummy_input)
        
        # Traitement identique à ton "Etape D" pour aligner géométriquement les tenseurs
        b, _, c = output_timm_raw.shape
        h = w = 518 // 14
        
        output_timm = output_timm_raw[:, 1:, :] # Retrait du CLS Token
        output_timm = output_timm.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    # 5. Validation
    print("MyModel Output shape :", output.shape)
    print("Timm Output shape    :", output_timm.shape)
    
    # L'erreur maximale absolue (tolérance normale : < 1e-5 à cause des flottants)
    max_diff = (output - output_timm).abs().max().item()
    print("Max absolute diff    :", max_diff)
    
    print("\nLancement de la visualisation pour dog.jpg...")
    visualize_attention(mymodel, "dog.jpg", device)
        
        
        
        
        
        