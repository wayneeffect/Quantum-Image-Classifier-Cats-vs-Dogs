import io
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pipeline import QMLPipeline

app = FastAPI()

# Enable CORS so your frontend UI service doesn't block request validation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = QMLPipeline()

# Non-blocking asynchronous HTTP client with built-in connection pooling
async_http_client = httpx.AsyncClient(timeout=45.0)

async def call_quantum_oracle_async(hamiltonian: list[list[float]], max_retries: int = 3, backoff_factor: int = 2) -> dict:
    """Dispatches the Hamiltonian matrix asynchronously with non-blocking backoff intervals."""
    url = "https://grok-wayne-s-quantum-algorithm.onrender.com/hybrid_vqe_qaoa"
    payload = {"hamiltonian": hamiltonian, "parameters": [0.35, 0.72, 0.15]}
    
    for attempt in range(max_retries):
        try:
            response = await async_http_client.post(url, json=payload)
            
            # Non-blocking sleep if rate limited (429), keeping the worker process alive
            if response.status_code == 429:
                sleep_time = backoff_factor ** attempt
                print(f"[Rate Limited] 429 Error. Asynchronously backing off for {sleep_time}s...")
                await asyncio.sleep(sleep_time)
                continue
                
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail=f"External Quantum Oracle returned failure code {response.status_code}")
                
            return response.json()
            
        except httpx.RequestError as exc:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=503, detail=f"Failed to communicate with Oracle infrastructure: {exc}")
                
    raise HTTPException(status_code=429, detail="Exceeded maximum automated quantum oracle dispatch retry windows.")

@app.get("/")
def health_check():
    return {"status": "healthy", "pipeline": "Asynchronous Quantum Image Classifier"}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        try:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file stream is unreadable or corrupted.")

        # Run feature map allocations
        raw_features = pipeline.extract_classical_features(img)
        hamiltonian = pipeline.map_to_quantum_hamiltonian(raw_features, size=4)
        
        # Call the non-blocking retry loop 
        oracle_data = await call_quantum_oracle_async(hamiltonian)
        return oracle_data

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        # Check if the process crashed due to RAM constraints
        print(f"[CRITICAL ERROR ENCOUNTERED]: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Core Pipeline Error: {str(e)}")
