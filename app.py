from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import QMLPipeline

app = FastAPI()
pipeline = QMLPipeline()

class ImagePayload(BaseModel):
    image_url: str  # Or accept raw bytes depending on your front-end

@app.get("/")
def health_check():
    return {"status": "healthy", "pipeline": "Quantum Image Classifier"}

@app.post("/classify")
def classify_image(payload: ImagePayload):
    try:
        # 1. Download/load image from payload URL
        img = pipeline.validate_and_load_image(payload.image_url)
        # 2. Extract features
        raw_features = pipeline.extract_classical_features(img)
        # 3. Reduce and shape matrix
        hamiltonian = pipeline.map_to_quantum_hamiltonian(raw_features, size=4)
        # 4. Call your external quantum oracle
        oracle_data = pipeline.call_quantum_oracle(hamiltonian)
        
        return oracle_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
