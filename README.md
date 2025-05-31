# Mtailor ONNX Flask App

This project hosts a Flask application that serves an ONNX model for image classification using a ResNet18 architecture.

---

## 📁 Project Files

- `app.py`               — Flask application entry point  
- `model.py`             — ONNX model loading and inference code  
- `convert_to_onnx.py`   — Script to convert PyTorch model to ONNX format  
- `pytorch_model.py`     — PyTorch model script  
- `Dockerfile`           — Docker configuration file  
- `mtailor_resnet18.onnx` — Pretrained ONNX model file  
- `n01440764_tench.jpeg` — Sample image 1  
- `n01667114_mud_turtle.JPEG` — Sample image 2  
- `test.py`              — Test script  
- `test_server.py`       — Server testing script  
- `resnet18-f37072fd.pth` — PyTorch weights file  

---

## 🚀 How to Run the Project

This guide includes instructions for running the project both on native Linux/Windows and using **Windows Subsystem for Linux (WSL)**.

---

### 1. Setup Environment (Windows + WSL)

#### A. Installing WSL on Windows

If you are on Windows 10 or 11, it's recommended to use WSL2 to get a Linux environment with full system call compatibility.

- Open PowerShell as Administrator and run:

```powershell
wsl --install
````

This will install the default Ubuntu distribution.
If you already have WSL installed, update it to WSL2:

```powershell
wsl --set-default-version 2
```

* Launch Ubuntu from your Start menu.

#### B. Updating and Installing Dependencies inside WSL

Once inside the WSL terminal, update package lists and install Python3, pip, and other essentials:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git docker.io
```

> **Note:** To use Docker inside WSL, install Docker Desktop for Windows and enable WSL integration in Docker Desktop settings.

---

### 2. Clone the Repository

Inside WSL terminal or native Linux terminal:

```bash
git clone (https://github.com/ZayanRashid295/mtailor_mlops_assignment)
```

---

### 3. Create and Activate Virtual Environment (Recommended)

Create a Python virtual environment to isolate dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Python Dependencies

If you have a `requirements.txt` file, install all dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, create it with the following content:

```text
flask
onnxruntime
torch
torchvision
numpy
pillow
```

Then run:

```bash
pip install flask onnxruntime torch torchvision numpy pillow
```

---

### 5. (Optional) Generate `requirements.txt`

If you installed packages manually and want to save them:

```bash
pip freeze > requirements.txt
```

---

### 6. Running Flask App Locally (without Docker)

Start the Flask app by running:

```bash
export FLASK_APP=app.py
export FLASK_ENV=development  # optional, for debug mode
flask run --host=0.0.0.0 --port=8080
```

The app will be available at [http://localhost:8080/](http://localhost:8080/) or [http://127.0.0.1:8080/](http://127.0.0.1:8080/).

---

### 7. Using Docker (Recommended for Deployment)

#### A. Install Docker

* **Windows:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and enable WSL 2 integration.
* **Linux:** Install Docker using your distro’s package manager.

Verify Docker installation:

```bash
docker --version
```

---

#### B. Build Docker Image

From the project root directory (inside WSL or Linux terminal):

```bash
docker build -t mtailor-onnx-app .
```

Make sure your `Dockerfile` and `requirements.txt` are in the root directory.

---

#### C. Run Docker Container

Run the container with port mapping:

```bash
docker run -p 8080:8080 mtailor-onnx-app
```

Visit [http://127.0.0.1:8080/](http://127.0.0.1:8080/) to access the app.

---

## 8. Using the API

### Endpoint:

```
POST http://127.0.0.1:8080/api/predict
```

### Payload:

Send an image file (e.g., JPEG, PNG) as form-data with key `file`.

Example using `curl`:

```bash
curl -X POST -F file=@path/to/image.jpg http://127.0.0.1:8080/api/predict
```

You will receive a JSON response with classification results.

---

## 🛠 Troubleshooting

* **Docker build error: `requirements.txt` not found**
  Make sure `requirements.txt` is in the same directory as the `Dockerfile`.

* **Docker daemon not running**

  * On Windows: Ensure Docker Desktop is running and WSL integration is enabled.
  * On Linux: Start Docker service with `sudo systemctl start docker`.

* **Permission denied running Docker in WSL**
  Add your user to the docker group:

  ```bash
  sudo usermod -aG docker $USER
  newgrp docker
  ```

* **Port 8080 already in use**
  Change the port in the Docker run command and Flask app accordingly:

  ```bash
  docker run -p 9090:8080 mtailor-onnx-app
  ```

* **Python package installation issues**
  Ensure Python, pip, and wheel packages are updated:

  ```bash
  pip install --upgrade pip setuptools wheel
  ```

## Files

1. **convert\_to\_onnx.py** — converting the PyTorch model to ONNX
2. **model.py** — modular ONNX model loader + inference + preprocessing classes
3. **Dockerfile** — a clean Dockerfile for Cerebrium deployment with Python dependencies
4. **app.py** — FastAPI server for inference inside the container
5. **test\_server.py** — test script to hit the deployed API remotely
6. **README.md** — outline of instructions for running everything

---

### 1. convert\_to\_onnx.py (PyTorch → ONNX conversion)

```
import torch
from pytorch_model import MyModel  # assuming model class name from pytorch_model.py

def convert_to_onnx():
    model = MyModel()
    model.load_state_dict(torch.load('pytorch_model_weights.pth', map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)  # batch size 1, 3 RGB channels, 224x224
    onnx_path = 'model.onnx'

    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    convert_to_onnx()
```

---

### 2. model.py (ONNX loader + preprocessor)

```
import numpy as np
import onnxruntime as ort
from PIL import Image

class ImagePreprocessor:
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def preprocess(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC to CHW
        img_np = np.expand_dims(img_np, axis=0)   # batch dim
        return img_np

class ONNXModel:
    def __init__(self, model_path='model.onnx'):
        self.session = ort.InferenceSession(model_path)

    def predict(self, input_tensor):
        inputs = {self.session.get_inputs()[0].name: input_tensor}
        outputs = self.session.run(None, inputs)
        return outputs[0]
```

---

### 3. Dockerfile (for Cerebrium deployment)

```Dockerfile
# Use an official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Run Flask app
CMD ["python", "app.py"]
```

---

### requirements.txt

```
flask
onnxruntime
fastapi
uvicorn
numpy
pillow
requests
```

---

### 4. app.py (FastAPI server for inference)

```python
from fastapi import FastAPI, UploadFile, File
from model import ONNXModel, ImagePreprocessor
import numpy as np

app = FastAPI()
model = ONNXModel()
preprocessor = ImagePreprocessor()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.jpg", "wb") as f:
        f.write(contents)

    input_tensor = preprocessor.preprocess("temp.jpg")
    preds = model.predict(input_tensor)
    top_class = int(np.argmax(preds[0]))
    top_prob = float(np.max(preds[0]))

    return {"class_id": top_class, "probability": top_prob}
```

---

### 5. test_server.py (test deployed API)

```python
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8080/api/predict" 

def send_image(image_path: str):
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(API_URL, files=files)
    return response.json()

if __name__ == "__main__":
    for img_path in ["n01440764_tench.jpeg", "n01667114_mud_turtle.JPEG"]:
        result = send_image(img_path)
        print(f"Image: {img_path}, Prediction: {result}")

```

---

pip install -r requirements.txt

```
1. Convert PyTorch model to ONNX:
```

python convert_to_onnx.py

```
2. Run local tests:
```

python test.py

```
3. Build Docker image:
```

docker build -t mtailor-onnx-model .

```

4. Push image to Docker Hub and deploy on Cerebrium (follow platform instructions)

5. Test remote deployment:
```

python test\_server.py --image-path \<path\_to\_image>

```

---

## File Descriptions

- `convert_to_onnx.py`: Converts PyTorch model to ONNX format
- `model.py`: Contains classes for image preprocessing and ONNX model inference
- `app.py`: FastAPI server running the model for inference
- `test.py`: Local tests for model correctness
- `test_server.py`: Test remote deployed model via API call
- `Dockerfile`: Docker setup for deployment
```

---


---


