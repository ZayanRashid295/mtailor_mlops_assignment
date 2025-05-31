import torch
from pytorch_model import Classifier, BasicBlock
from PIL import Image

def convert_to_onnx(model_path: str, onnx_path: str):
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Dummy input matching model input size (batch, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved to {onnx_path}")


if __name__ == "__main__":
    convert_to_onnx(
        model_path="./resnet18-f37072fd.pth",
        onnx_path="./mtailor_resnet18.onnx",
    )
