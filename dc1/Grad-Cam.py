from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from torchvision import models

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
target_layers = model.features[-1]

model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

# Updating the classifier layer in accordance to our 6 classes
model.classifier[1] = torch.nn.Linear(model.last_channel, 6)

# Load the trained model weights (temporarily required to update the path manually)
model.load_state_dict(torch.load('MobilNet_.pth'))

model.eval()
cam = GradCAM(model=model, target_layers=target_layers)

data_path = Path(__file__).parent.parent
test_x_path = data_path / "data/X_test.npy"
test = np.load(test_x_path)

# Select the index of an image you want to be displayed
image_to_display = np.squeeze(test[10])

img = Image.fromarray(image_to_display)

# Normalize the data
rgb_img = np.array(img.convert("RGB")) / 255

# Apply the same preprocessing transforms as done during model training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

])

rgb_img = transform(img).cpu().permute(1,2,0).numpy()
input_tensor = transform(img).unsqueeze(0)

# Move the tensor to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)

# Generate the Grad-CAM heatmap
targets = None
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
plt.axis("off")

# Display image
plt.show()
print(dir(cam.device))

