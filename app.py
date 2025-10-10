from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import torch
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

# Local imports
from src.models import UNet
from src.data import preprocess_single_image
from src.utils import calculate_affected_percentage


# App & Config
app = FastAPI(title="Fundus Segmentation API")

CHECKPOINT_DIR = "outputs/checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Request & Response Models
class ModelRequest(BaseModel):
    filename: str


class ModelInfo(BaseModel):
    type: str
    keys_count: Optional[int]
    keys_sample: Optional[List[str]]


class ModelResponse(BaseModel):
    filename: str
    info: ModelInfo


class PredictionResult(BaseModel):
    filename: str
    affected_percentage: float
    mask_preview: Optional[str]  # base64-encoded PNG


# Utils
def load_checkpoint(model_path: str) -> Dict[str, Any]:
    """Load checkpoint safely from disk."""
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    return torch.load(model_path, map_location=DEVICE)


def build_unet(config: Dict[str, Any]) -> UNet:
    """Build UNet from config dictionary."""
    return UNet(
        n_channels=config.get("n_channels", 3),
        n_classes=config.get("n_classes", 2),
        bilinear=config.get("bilinear", True),
    )


def create_mask_preview(image_array: np.ndarray, pred_mask: np.ndarray) -> str:
    """Overlay segmentation mask on image and return base64-encoded PNG."""
    overlay = image_array.copy()
    colored_mask = np.zeros_like(overlay)
    colored_mask[pred_mask > 0.5] = [255, 0, 0]  # red overlay
    combined = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

    _, buffer = cv2.imencode(".png", combined)
    return base64.b64encode(buffer).decode("utf-8")


# API Endpoints
@app.get("/health")
def healthcheck():
    """Healthcheck endpoint."""
    return {"status": "ok"}


@app.post("/get-model", response_model=ModelResponse)
def get_model(request: ModelRequest):
    """Inspect checkpoint file and return metadata."""
    file_path = os.path.join(CHECKPOINT_DIR, request.filename)

    checkpoint = load_checkpoint(file_path)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        info = ModelInfo(
            type="full_checkpoint",
            keys_count=len(checkpoint["state_dict"].keys()),
            keys_sample=list(checkpoint["state_dict"].keys())[:5],
        )
    elif isinstance(checkpoint, dict):
        info = ModelInfo(
            type="state_dict_only",
            keys_count=len(checkpoint.keys()),
            keys_sample=list(checkpoint.keys())[:5],
        )
    else:
        info = ModelInfo(type="unknown")

    return ModelResponse(filename=request.filename, info=info)


@app.post("/inference", response_model=PredictionResult)
async def inference(input_file: UploadFile = File(...), model_filename: str = Form(...)):
    """Run inference on a single fundus image."""
    # Load checkpoint
    model_path = os.path.join(CHECKPOINT_DIR, model_filename)
    checkpoint = load_checkpoint(model_path)

    # Initialize model
    config = checkpoint.get("model_config", {"n_channels": 3, "n_classes": 2, "bilinear": True})
    model = build_unet(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    # Preprocess input image
    try:
        image_bytes = await input_file.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(pil_image)

        image_tensor = preprocess_single_image(image_array, config.get("image_size", 512))
        image_tensor = image_tensor.to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read or preprocess image: {str(e)}")

    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_mask = (probs[0, 1, :, :] > 0.5).float().cpu().numpy()

    # Postprocess results
    affected_percentage = calculate_affected_percentage(pred_mask)
    mask_preview_b64 = create_mask_preview(image_array, pred_mask)

    return PredictionResult(
        filename=input_file.filename,
        affected_percentage=float(affected_percentage),
        mask_preview=mask_preview_b64,
    )
