from flask import Flask, request, jsonify
from PIL import Image
import io
from model import ONNXModel, ImagePreprocessor

app = Flask(__name__)

model = ONNXModel("./mtailor_resnet18.onnx")
preprocessor = ImagePreprocessor()

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    input_tensor = preprocessor.preprocess(img)
    output = model.predict(input_tensor)
    pred_class = int(output.argmax(axis=1)[0])
    return jsonify({"predicted_class": pred_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
