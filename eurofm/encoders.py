"""Registry of frozen image encoders, loaded through timm or open_clip.

Every encoder is wrapped in :class:`Encoder`, which takes raw uint8 EuroSAT tiles
(N, 64, 64, 3), upsamples and normalises them on the GPU, and returns pooled features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

ENCODERS: dict[str, dict] = {
    # --- classic transfer learning baseline ---
    "resnet50_in1k": dict(lib="timm", name="resnet50.a1_in1k", desc="ResNet-50, ImageNet-1k supervised (2015-era baseline)"),
    # --- self-supervised vision foundation models (Meta DINOv3, 2025) ---
    "dinov3_vitb16_web": dict(lib="timm", name="vit_base_patch16_dinov3.lvd1689m", desc="DINOv3 ViT-B/16, 1.7B web images"),
    "dinov3_vitl16_web": dict(lib="timm", name="vit_large_patch16_dinov3.lvd1689m", desc="DINOv3 ViT-L/16, 1.7B web images"),
    "dinov3_vitl16_sat": dict(lib="timm", name="vit_large_patch16_dinov3.sat493m", desc="DINOv3 ViT-L/16, 493M satellite tiles"),
    # --- image-text models (support zero-shot classification) ---
    "clip_vitb16": dict(lib="open_clip", name="ViT-B-16", pretrained="openai", desc="OpenAI CLIP ViT-B/16"),
    "siglip2_vitb16": dict(lib="open_clip", name="ViT-B-16-SigLIP2", pretrained="webli", desc="Google SigLIP 2 ViT-B/16"),
    "remoteclip_vitb32": dict(
        lib="open_clip",
        name="ViT-B-32",
        hf_ckpt=("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt"),
        desc="RemoteCLIP ViT-B/32 (CLIP tuned on remote-sensing captions)",
    ),
}

IMAGE_SIZE = 224


@dataclass
class Encoder:
    key: str
    model: torch.nn.Module
    lib: str
    mean: tuple
    std: tuple
    image_size: int
    feature_dim: int
    tokenizer: object | None = None  # open_clip only
    info: dict = field(default_factory=dict)

    @property
    def has_text(self) -> bool:
        return self.tokenizer is not None

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 NHWC (any device) -> normalised float NCHW at the encoder's resolution."""
        x = images.permute(0, 3, 1, 2).float().div_(255.0)
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
            x = x.clamp_(0.0, 1.0)
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised NCHW -> pooled features (differentiable; used by LoRA fine-tuning)."""
        if self.lib == "open_clip":
            return self.model.encode_image(x)
        return self.model(x)

    @torch.no_grad()
    def encode_images(self, images: np.ndarray, batch_size: int = 128, amp: bool = True) -> np.ndarray:
        dev = next(self.model.parameters()).device
        self.model.eval()
        out = []
        for i in range(0, len(images), batch_size):
            batch = torch.from_numpy(np.array(images[i : i + batch_size])).to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
                feats = self.forward_features(self.preprocess(batch))
            out.append(feats.float().cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        tokens = self.tokenizer(texts).to(dev)
        return self.model.encode_text(tokens).float()


def load_encoder(key: str, pretrained: bool = True, device: torch.device | str = "cpu") -> Encoder:
    """Load an encoder from the registry. ``pretrained=False`` gives random weights (offline smoke tests)."""
    if key not in ENCODERS:
        raise KeyError(f"Unknown encoder {key!r}. Choose from: {', '.join(ENCODERS)}")
    spec = ENCODERS[key]

    if spec["lib"] == "timm":
        import timm

        model = timm.create_model(spec["name"], pretrained=pretrained, num_classes=0)
        cfg = timm.models.get_pretrained_cfg(spec["name"])  # carries e.g. the satellite-specific mean/std
        enc = Encoder(key, model, "timm", tuple(cfg.mean), tuple(cfg.std), IMAGE_SIZE, model.num_features)

    elif spec["lib"] == "open_clip":
        import open_clip

        tag = spec.get("pretrained") if pretrained else None
        model = open_clip.create_model(spec["name"], pretrained=tag)
        if pretrained and "hf_ckpt" in spec:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(*spec["hf_ckpt"])
            state = torch.load(path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(f"{key}: checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
        cfg = model.visual.preprocess_cfg
        size = cfg["size"][0] if isinstance(cfg["size"], (tuple, list)) else cfg["size"]
        tokenizer = open_clip.get_tokenizer(spec["name"])
        with torch.no_grad():
            dim = model.encode_image(torch.zeros(1, 3, size, size)).shape[-1]
        enc = Encoder(key, model, "open_clip", tuple(cfg["mean"]), tuple(cfg["std"]), int(size), int(dim), tokenizer)
    else:
        raise ValueError(spec["lib"])

    enc.info = dict(spec)
    enc.model.to(device).eval()
    return enc
