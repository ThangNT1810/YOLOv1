import onnx
import onnxruntime
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from test import transform, cellboxes_to_boxes, non_max_suppression, Yolov1
import torch
import torch.onnx
import os

#Create ONNX
checkpoint_path = '/content/drive/MyDrive/YOLOv1/my_checkpoint.pth.tar'

# Load your trained YOLOv1 model (replace with your actual model loading code)
if os.path.exists(checkpoint_path):
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20)  # Tạo mô hình YOLO của bạn ở đây
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

# Create a dummy input tensor (replace with actual input size)
dummy_input = torch.randn(1, 3, 448, 448)

# Export the model to ONNX
onnx_model_path = "yolov1.onnx"
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(model, dummy_input, onnx_model_path, input_names=input_names, output_names=output_names)


#Test ONNX
# Load YOLOv1 ONNX model
onnx_model_path = "yolov1.onnx"
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

# Load the image
image_path = '/content/drive/MyDrive/YOLOv1/image_test/image.jpg'
image = Image.open(image_path)

# Transform and preprocess the image
transformed_image, _ = transform(image, [])  # Make sure the transform function is defined
image_tensor = transformed_image.unsqueeze(0)

# Perform prediction using the ONNX model
with torch.no_grad():
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_outputs = onnx_session.run([output_name], {input_name: image_tensor.numpy()})
    bboxes = cellboxes_to_boxes(torch.tensor(onnx_outputs[0]), S=7)
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")

# Convert the image to a NumPy array
im = np.array(image)

# Get the image dimensions
height, width, _ = im.shape

# Draw bounding boxes on the image
for box in bboxes:
    if len(box) == 6:
        class_id, confidence, x_center, y_center, box_width, box_height = box
        x_min = int((x_center - box_width / 2) * width)
        y_min = int((y_center - box_height / 2) * height)
        x_max = int((x_center + box_width / 2) * width)
        y_max = int((y_center + box_height / 2) * height)
        cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

# Display the image with bounding boxes using matplotlib
plt.imshow(im)
plt.show()
