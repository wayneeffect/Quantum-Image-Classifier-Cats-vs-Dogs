# 🐈‍⬛ Quantum Image Classifier: Cats vs. Dogs 🐕

A hybrid **Quantum Machine Learning (QML)** desktop pipeline that extracts classical high-level image features using a Convolutional Neural Network (CNN) and maps them to a remote cloud-based **Quantum Oracle** to perform binary classification. 

This repository serves as a practical implementation of hybrid classical-quantum variational workflows, utilizing classical feature reduction techniques paired with an external, API-driven Variational Quantum Eigensolver (VQE) / Quantum Approximate Optimization Algorithm (QAOA) oracle architecture.

---

## 🗺️ System Architecture

The application implements a strict three-stage hybrid data pipeline:

1. **Classical Feature Extraction:** High-resolution input images are validated and passed through a pre-trained classical network (**MobileNetV2**). The model strips the final classification layer, capturing deep spatial hierarchies as a compressed 1D feature map vector.
2. **Hermitian Matrix Mapping:** The feature vector undergoes Dimensionality Reduction (truncation and Min-Max scaling between -1 and 1). It is then reshaped and structured into a symmetric square grid ($4 \times 4$), satisfying the Hermitian properties ($H = H^\dagger$) required by quantum operators.
3. **Quantum Oracle Execution:** The matrix is transmitted via HTTP POST as a Hamiltonian payload to the remote **Grok & Wayne Quantum Oracle API**. The oracle processes the landscape on a quantum simulator/QPU ansatz and returns an optimal expectation value (energy state). A static threshold maps this energy to a final prediction: `CAT` or `DOG`.

---

## 🚀 Features (MoSCoW Framework)

* **Must Have:** End-to-end vision-to-quantum execution pipeline, zero hardcoded endpoints via isolated `.env` environments, strict image input sanitization and verification, and automated handling for network latency/cold-start timeouts.
* **Should Have:** Type hinting throughout core modules, comprehensive error handling for API failures, and isolated configuration validation interfaces.
* **Could Have:** Local feature extraction caching (SQLite/NumPy) and a standalone Streamlit user visualization dashboard.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch & Torchvision (Classical Computer Vision)
* **Image Processing:** Pillow (PIL)
* **Data Manipulation:** NumPy
* **Networking:** Requests (HTTP/JSON REST communications)

---

## 📦 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/quantum-cat-dog-classifier.git](https://github.com/YOUR_USERNAME/quantum-cat-dog-classifier.git)
   cd quantum-cat-dog-classifier
