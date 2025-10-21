import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import requests

# --- Пути к весам разных стилей (.pt) ---
WEIGHTS_DICT = {
    "face_paint": "models/face_paint_512_v2.pt",
    "hayao": "models/celeba_distill.pt",
    "paprika": "models/paprika.pt"
}

# --- Ссылки на скачивание весов (Hugging Face) ---
MODEL_URLS = {
    "face_paint": "https://huggingface.co/lllyasviel/AnimeGANv2/resolve/main/face_paint_512_v2.pt",
    "hayao": "https://huggingface.co/lllyasviel/AnimeGANv2/resolve/main/celeba_distill.pt",
    "paprika": "https://huggingface.co/lllyasviel/AnimeGANv2/resolve/main/paprika.pt"
}


class AnimeGANv2Cartoonizer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # модель создается только при load()

    def load(self, style: str = "face_paint"):
        """Load model weights for the selected style"""
        if style not in WEIGHTS_DICT:
            raise ValueError(f"Unknown style '{style}', choose from {list(WEIGHTS_DICT.keys())}")

        model_path = WEIGHTS_DICT[style]

        # Скачать веса, если их нет
        if not os.path.exists(model_path):
            self._download_model(style)

        # Инициализация модели генератора
        self.model = torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "generator",
            pretrained=False
        )

        # Загрузка весов
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    def _download_model(self, style):
        url = MODEL_URLS[style]
        os.makedirs("models", exist_ok=True)
        print(f"Downloading '{style}' weights from Hugging Face...")
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download weights: HTTP {r.status_code}")
        with open(WEIGHTS_DICT[style], "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Weights '{style}' downloaded successfully.")

    def cartoonify(self, image: Image.Image):
        """Convert input image to cartoon style using AnimeGANv2."""
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call load(style) first.")

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        with torch.no_grad():
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            output_tensor = self.model(input_tensor)[0].cpu()
            output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)
            cartoon_image = transforms.ToPILImage()(output_tensor)
            return cartoon_image
