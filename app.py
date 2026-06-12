from fastapi import FastAPI, HTTPException, File, UploadFile
from pipeline import QMLPipeline
from PIL import Image
import io

app = FastAPI()
pipeline = QMLPipeline()

@app.get("/")
def health_check():
    return {"status": "healthy", "pipeline": "Quantum Image Classifier"}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        # 1. Read the raw bytes directly from the Streamlit upload stream
        contents = await file.read()
        
        # 2. Convert raw bytes into a PIL Image instance
        try:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid or supported image format.")

        # 3. Process via your existing QMLPipeline methods
        # (Bypassing validate_and_load_image since it's already in memory as a PIL Image)
        raw_features = pipeline.extract_classical_features(img)
        
        # 4. Map down to the 4x4 matrix representation
        hamiltonian = pipeline.map_to_quantum_hamiltonian(raw_features, size=4)
        
        # 5. Connect to the external Grok & Wayne Oracle
        oracle_data = pipeline.call_quantum_oracle(hamiltonian)
        
        return oracle_data

    except Exception as e:
        # This catches any other internal processing errors and relays the details
        raise HTTPException(status_code=500, detail=str(e))
