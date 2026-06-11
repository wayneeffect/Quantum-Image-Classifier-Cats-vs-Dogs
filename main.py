import sys
from config import Config
from pipeline import QMLPipeline

def run_classification(image_path: str) -> None:
    # 1. Initialize configuration validation
    try:
        Config.validate()
    except ValueError as config_err:
        print(f"[CONFIGURATION ERROR] {config_err}")
        sys.exit(1)

    # 2. Instantiate pipeline
    print("[Pipeline] Initializing classical vision models...")
    pipeline = QMLPipeline()

    # 3. Load and Validate Image Input
    try:
        print(f"[Pipeline] Loading image: {image_path}")
        img = pipeline.validate_and_load_image(image_path)
    except (FileNotFoundError, ValueError) as input_err:
        print(f"[INPUT ERROR] {input_err}")
        sys.exit(1)

    # 4. Feature Extraction & Hybrid Mapping
    print("[Pipeline] Computing classical feature maps via CNN...")
    raw_features = pipeline.extract_classical_features(img)
    
    print(f"[Pipeline] Mapping features to a {Config.MATRIX_SIZE}x{Config.MATRIX_SIZE} Hermitian matrix...")
    hamiltonian = pipeline.map_to_quantum_hamiltonian(raw_features, Config.MATRIX_SIZE)

    # 5. Connect with Quantum Oracle API
    print("[Pipeline] Dispatching payload to Quantum Oracle via HTTP POST...")
    try:
        oracle_data = pipeline.call_quantum_oracle(hamiltonian)
    except Exception as network_err:
        print(f"\n[ORACLE COMMUNICATION ERROR] {network_err}")
        print("Suggestion: If the application timed out, wait 1-2 minutes and retry to let Render wake up.")
        sys.exit(1)

    # 6. Parse Response & Map Binary Decisions
    print("\n" + "="*40)
    print("      QUANTUM ORACLE RESPONSE MATRIX      ")
    print("="*40)
    
    # Extract expectation values/optimal energy from response
    energy = oracle_data.get("optimal_energy", 0.0)
    print(f"Calculated Optimal Energy (Expectation): {energy}")
    
    # Binary classification mapping logic via configured threshold
    if energy > Config.THRESHOLD:
        label = "DOG"
    else:
        label = "CAT"
        
    print(f"Classification Decision Boundary: > {Config.THRESHOLD} -> DOG")
    print(f"FINAL PREDICTION RESULT: {label}")
    print("="*40)

if __name__ == "__main__":
    # Supply a default fallback sample if no command line argument is passed
    target_image = sys.argv[1] if len(sys.argv) > 1 else "sample_pet.jpg"
    
    print("--- Starting Hybrid Quantum Machine Learning Pipeline ---")
    run_classification(target_image)
