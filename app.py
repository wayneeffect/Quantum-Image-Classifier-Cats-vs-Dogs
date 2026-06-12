import io
import time
from fastapi import FastAPI, HTTPException, File, UploadFile
from PIL import Image
import requests
from pipeline import QMLPipeline

app = FastAPI()
pipeline = QMLPipeline()


def call_quantum_oracle_with_retry(
    hamiltonian: list[list[float]], max_retries: int = 3, backoff_factor: int = 2
) -> dict:
    """Dispatches the Hamiltonian matrix to the external oracle with exponential backoff handling

    for HTTP 429 Rate Limiting.
    """
    url = "https://grok-wayne-s-quantum-algorithm.onrender.com/hybrid_vqe_qaoa"
    payload = {"hamiltonian": hamiltonian, "parameters": [0.35, 0.72, 0.15]}
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            # If rate limited (429), calculate delay, pause execution, and loop back
            if response.status_code == 429:
                sleep_time = backoff_factor**attempt
                print(
                    f"[Rate Limited] Status 429 encountered. Retrying attempt {attempt + 1}/{max_retries} in {sleep_time}s..."
                )
                time.sleep(sleep_time)
                continue

            if response.status_code != 200:
                raise requests.exceptions.HTTPError(
                    f"Oracle Error: {response.status_code}"
                )

            return response.json()

        except requests.exceptions.RequestException as e:
            # If it's our last attempt and we hit a network glitch, raise the exception up
            if attempt == max_retries - 1:
                raise e

    raise requests.exceptions.RequestException(
        "Failed to complete request after maximum rate-limit retries."
    )


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
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid or supported image format.",
            )

        # 3. Process via your existing QMLPipeline methods
        raw_features = pipeline.extract_classical_features(img)

        # 4. Map down to the 4x4 matrix representation
        hamiltonian = pipeline.map_to_quantum_hamiltonian(raw_features, size=4)

        # 5. Connect to the external Grok & Wayne Oracle using the retry safety wrapper
        oracle_data = call_quantum_oracle_with_retry(
            hamiltonian, max_retries=3, backoff_factor=2
        )

        return oracle_data

    except Exception as e:
        # Catch internal pipeline exceptions or final network failures and send them to the UI
        raise HTTPException(status_code=500, detail=str(e))
