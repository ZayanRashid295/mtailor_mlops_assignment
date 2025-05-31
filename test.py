from model import ONNXModel, ImagePreprocessor
from PIL import Image

def test_local():
    model = ONNXModel("./mtailor_resnet18.onnx")
    preprocessor = ImagePreprocessor()

    test_images = [
        ("n01440764_tench.jpeg", 0),
        ("n01667114_mud_turtle.JPEG", 35),
    ]

    for filename, expected_class in test_images:
        img = Image.open(filename)
        input_tensor = preprocessor.preprocess(img)
        output = model.predict(input_tensor)
        pred_class = output.argmax(axis=1)[0]
        print(f"{filename}: predicted={pred_class}, expected={expected_class}")
        assert pred_class == expected_class, f"Mismatch for {filename}"

if __name__ == "__main__":
    test_local()
