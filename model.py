import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img_t = self.transforms(img)
        # Convert to numpy and add batch dimension
        return img_t.unsqueeze(0).cpu().numpy().astype(np.float32)


class ONNXModel:
    def __init__(self, onnx_model_path: str):
        self.session = onnxruntime.InferenceSession(onnx_model_path)

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        inputs = {self.session.get_inputs()[0].name: input_tensor}
        outputs = self.session.run(None, inputs)
        return outputs[0]  # logits or probabilities


if __name__ == "__main__":
    # Simple test
    preprocessor = ImagePreprocessor()
    model = ONNXModel("./mtailor_resnet18.onnx")

    img = Image.open("./n01667114_mud_turtle.JPEG")
    input_tensor = preprocessor.preprocess(img)
    output = model.predict(input_tensor)

    print("Predicted class:", output.argmax(axis=1))
