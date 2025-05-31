import requests
from PIL import Image
import io

API_URL = "https://your-deployment-url/api/predict"  # Change this

def send_image(image_path: str):
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(API_URL, files=files)
    return response.json()

if __name__ == "__main__":
    for img_path in ["n01440764_tench.jpeg", "n01667114_mud_turtle.JPEG"]:
        result = send_image(img_path)
        print(f"Image: {img_path}, Prediction: {result}")
