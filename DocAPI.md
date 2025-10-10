# Fundus Segmentation API

This project provides FastAPI-based API endpoints for checking system health, getting model info, and running inference on a single file.

### 1. Install dependencies
Make sure you have Python 3.7+ installed.

```bash
pip install -r requirements.txt
```

### 2. Run the local host
Start the FastAPI server with:
```bash
uvicorn app.main:app --reload
```

The API will be available at:
```
http://127.0.0.1:8000
```

The API doc will be available at:
```
http://127.0.0.1:8000/docs
```

### 3. API Endpoints
1. GET/health: healthcheck endpoint
Example usage:
```bash
curl -X GET http://127.0.0.1:8000/health
```

2. POST/get-model: Inspect checkpoint file and return metadata.
Example usage:
```bash
curl -X POST http://127.0.0.1:8000/get-model \
     -H "Content-Type: application/json" \
     -d '{"filename": "best-model.pth"}'
```

3. POST/inference: Run inference on a single fundus image.
Example usage:
```bash
curl -X POST http://127.0.0.1:8000/inference \
     -F "file=@test.png" \
     -F "model_filename=best-model.pth"

```