import torch
# print("Number of GPUs available:", torch.cuda.device_count())
# print("Current GPU device:", torch.cuda.current_device())
# print("GPU device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)