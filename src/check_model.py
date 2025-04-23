import onnx

print("Checking onnx")

new_path = "/workspace/svuet/Summarization/Foul_Detection/onnx_model/resnet18.onnx"
onnx_model = onnx.load(new_path)
# print(onnx_model)
status = onnx.checker.check_model(onnx_model)

print("Onnx is valid: ", status)