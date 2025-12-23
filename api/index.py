import io
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "yolo_dataset" / "train_chassi_detect2" / "weights" / "best.pt"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo não encontrado no caminho: {MODEL_PATH}")

model = YOLO(MODEL_PATH)  # carrega o modelo uma única vez

app = FastAPI(title="Detector de chassi", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir: Optional[Path] = (Path(__file__).resolve().parent / "static") if (
    Path(__file__).resolve().parent / "static"
).exists() else None

if static_dir:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_path": str(MODEL_PATH)}


@app.get("/")
async def serve_index():
    if static_dir:
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    return {"message": "Envie uma imagem via POST /predict"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Não foi possível abrir a imagem.") from exc

    start_time = time.perf_counter()
    results = model.predict(image, imgsz=1024, conf=0.25, verbose=False)
    inference_time = time.perf_counter() - start_time

    prediction = results[0]
    annotated = prediction.plot()  # numpy array em BGR
    annotated_image = Image.fromarray(annotated[:, :, ::-1])  # converte para RGB

    buffer = io.BytesIO()
    annotated_image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    headers = {
        "X-Inference-Time": f"{inference_time:.3f}",
        "X-Detections": str(len(prediction.boxes) if prediction and prediction.boxes is not None else 0),
        "X-Model-Path": str(MODEL_PATH),
    }

    return StreamingResponse(buffer, media_type="image/jpeg", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
