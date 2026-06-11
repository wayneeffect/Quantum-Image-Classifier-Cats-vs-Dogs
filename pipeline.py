import numpy as np
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from config import Config

class QMLPipeline:
    def __init__(self):
        # Using MobileNetV2 as an efficient feature extractor
        # weights=MobileNet_V2_Weights.DEFAULT is standard in modern PyTorch
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.eval()
        
        # Standard ImageNet pre-processing transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def validate_and_load_image(self, image_path: str) -> Image.Image:
        """Validates file existence and structure before loading it into memory."""
        try:
            img = Image.open(image_path)
            img.verify()  # Verify it's a valid, uncorrupted image file
            return Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Target image file not found at: {image_path}")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image format: {e}")

    def extract_classical_features(self, img: Image.Image) -> np.ndarray:
        """Compresses image data into a compact classical feature vector via a CNN."""
        tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            features = self.model.features(tensor)
            # Global Average Pooling flattens 2D feature maps to a 1D vector (1280 features)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            flattened = torch.flatten(pooled, 1)
            
        return flattened.numpy()[0]

    def map_to_quantum_hamiltonian(self, features: np.ndarray, size: int) -> list[list[float]]:
        """
        Reduces feature vector dimensions via truncation and normalization,
        mapping them to a symmetric/Hermitian grid for the quantum simulator.
        """
        required_elements = size * size
        truncated = features[:required_elements]
        
        # If the feature vector is too short, pad with zeros
        if len(truncated) < required_elements:
            truncated = np.pad(truncated, (0, required_elements - len(truncated)))
            
        # Min-Max Normalize features strictly between -1 and 1
        f_min, f_max = np.min(truncated), np.max(truncated)
        if f_max > f_min:
            truncated = 2 * ((truncated - f_min) / (f_max - f_min)) - 1
        else:
            truncated = np.zeros_like(truncated)

        # Reshape to a square matrix
        matrix = truncated.reshape((size, size))
        
        # Enforce Hermitian symmetry properties required by quantum states (H = H^T)
        hermitian_matrix = (matrix + matrix.T) / 2
        return hermitian_matrix.tolist()

    def call_quantum_oracle(self, hamiltonian: list[list[float]]) -> dict:
        """Executes a secure HTTP POST call to the remote Quantum Oracle."""
        payload = {
            "hamiltonian": hamiltonian,
            "parameters": [0.35, 0.72, 0.15]  # Static ansatz angles for VQE/QAOA setup
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                Config.ORACLE_API_URL, 
                json=payload, 
                headers=headers, 
                timeout=Config.ORACLE_TIMEOUT
            )
            
            # Handle standard HTTP error statuses explicitly
            if response.status_code != 200:
                raise requests.exceptions.HTTPError(
                    f"Oracle returned status code {response.status_code}: {response.text}"
                )
                
            return response.json()

        except requests.exceptions.Timeout:
            raise requests.exceptions.Timeout(
                f"The request to the quantum oracle timed out after {Config.ORACLE_TIMEOUT} seconds. "
                "The Render instance may be spinning up from a cold sleep."
            )
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Network transport layer failure: {e}")
