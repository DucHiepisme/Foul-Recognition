import argparse
import torch
import onnx
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components import Resnet18

def convert_pt2onnx(path_model: str):
    
    print("Converting model to onnx!")
    if ".pt" in path_model:
        new_path = path_model.replace(".pt", ".onnx")
    elif "ckpt" in path_model:
        new_path = path_model.replace(".ckpt", ".onnx")

    new_path = "onnx_model/resnet18.onnx"

    print("New path: ", new_path)

    if torch.cuda.is_available(): 
        device = "cuda:0"
    else:
        device = 'cpu'

    checkpoint = torch.load(path_model, map_location=device)

    # Check if it's a state_dict or full model
    if isinstance(checkpoint, dict):
        # Instantiate the model
        model = Resnet18()  # Define your model architecture
        # Load the state_dict
        state_dict = checkpoint.get('state_dict', checkpoint)  # Common key in Lightning checkpoints
        # Remove 'module.' prefix if trained with DataParallel
        state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
        print(state_dict)
        model.load_state_dict(state_dict)
    else:
        # If the checkpoint is the full model
        model = checkpoint

    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    # torch.onnx.export(
    #     model,                     # PyTorch model
    #     dummy_input,              # Sample input
    #     new_path,           # Output ONNX file
    #     # export_params=True,       # Export trained weights
    #     opset_version=12         # ONNX opset version (11 is widely compatible)
    # )

    print("Successly converted model to onnx!")
    print("Checking onnx")

    onnx_model = onnx.load(new_path)
    status = onnx.checker.check_model(onnx_model)

    print("Onnx is valid: ", status)

def main():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-m", "--model",
                  help="Path to ckpt model", default='logs/train/runs/2025-03-02_08-14-14/checkpoints/epoch_051.ckpt')

    args = parser.parse_args()
    
    convert_pt2onnx(path_model=args.model)

if __name__ == "__main__":
    main()