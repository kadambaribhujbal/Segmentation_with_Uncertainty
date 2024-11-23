# Aleotoric + Epistemic

import sys
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


# Add 'baseline' folder to Python path
sys.path.append('./baseline')

# Import tiramisu from the baseline folder
from tiramisu import FCDenseNet57_aleatoric
from utils.training import load_weights

# Paths
CAMVID_PATH = "./CamVid/CamVid/test"
# model_path = "./trained_weights/model_epoch_10_val_loss_batch1_0.5dp_lr1e-4_lrdecay0.995_combined.pth"
# model_path = "./trained_weights/combined_22ndnov_model_epoch_20_iou_33_5.pth"

model_path = "./trained_weights/23rdnov_combined_30epoch_iou_36_3.pth"

# model_path = "./trained_weights/model_epoch_10_val_loss_loss2_t10.pth"
# Preprocessing
normalize = transforms.Normalize(mean=[0.41189489566336, 0.4251328133025, 0.4326707089857],
                                  std=[0.27413549931506, 0.28506257482912, 0.28284674400252])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCDenseNet57_aleatoric(n_classes=12, dropout=0.5).to(device)
load_weights(model, model_path)
model.train()  # Switch to train mode for MC dropout

# Load the first image
image_path = sorted(Path(CAMVID_PATH).glob("*.png"))[0]  
input_image = Image.open(image_path).convert("RGB")
input_tensor = transform(input_image).unsqueeze(0).to(device)


# Monte Carlo sampling for epistemic uncertainty
mc_samples = 50
outputs_mc = []

with torch.no_grad():
    for _ in range(mc_samples):
        output, _ = model(input_tensor)  # Log-var is ignored for epistemic uncertainty
        probs = F.softmax(output, dim=1)  # Convert logits to probabilities
        outputs_mc.append(probs)

outputs_mc = torch.stack(outputs_mc)  # Shape: (mc_samples, batch_size, n_classes, H, W)
mean_output = outputs_mc.mean(dim=0)  # Mean of predictions
epistemic_uncertainty = outputs_mc.var(dim=0).mean(dim=1).squeeze(0).cpu().numpy()  # Variance mean over classes

# Get model output and aleatoric uncertainty
with torch.no_grad():
    model.eval()  # Switch back to eval mode
    output, log_var = model(input_tensor)

output_classes = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

# Decode the input image
def decode_image(tensor):
    """Decode normalized image tensor for visualization."""
    inp = tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.41189489566336, 0.4251328133025, 0.4326707089857])
    std = np.array([0.27413549931506, 0.28506257482912, 0.28284674400252])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

input_image_decoded = decode_image(input_tensor.squeeze(0).cpu())

# Aleatoric uncertainty visualization
def plot_with_uncertainty(input_tensor, uncertainty_map, mode="Uncertainty"):
    """Visualize input image with uncertainty overlay."""
    input_decoded = decode_image(input_tensor.squeeze(0).cpu())

    plt.imshow(input_decoded)
    plt.pcolor(uncertainty_map, alpha=0.5, cmap="viridis")
    plt.colorbar(label="Uncertainty")
    plt.title(mode)
    plt.axis("off")
    plt.show()

# Plot input, output, aleatoric and epistemic uncertainties
plt.figure(figsize=(24, 6))

# Input image
plt.subplot(1, 4, 1)
plt.imshow(input_image_decoded)
plt.title("Input Image")
plt.axis("off")

# Output segmentation
plt.subplot(1, 4, 2)
plt.imshow(output_classes, cmap="viridis")
plt.title("Model Output")
plt.axis("off")

# Aleatoric uncertainty
plt.subplot(1, 4, 3)
# aleatoric_uncertainty = log_var.squeeze(0).exp().mean(dim=0).cpu().numpy()  # Mean over classes
aleatoric_uncertainty = -log_var.squeeze(0).mean(dim=0).cpu().numpy()  # Mean over classes
plt.imshow(input_image_decoded)
plt.pcolor(aleatoric_uncertainty, alpha=0.5, cmap="viridis")
plt.colorbar(label="Aleatoric Uncertainty")
plt.title("Aleatoric Uncertainty")
plt.axis("off")


# Epistemic uncertainty
plt.subplot(1, 4, 4)
plt.imshow(input_image_decoded)
plt.pcolor(epistemic_uncertainty, alpha=0.5, cmap="viridis")
plt.colorbar(label="Epistemic Uncertainty")
plt.title("Epistemic Uncertainty")
plt.axis("off")

print(f"Aleatoric uncertainty shape: {aleatoric_uncertainty.shape}")
print(f"Epistemic uncertainty shape: {epistemic_uncertainty.shape}")
print(f"Input image shape: {input_image_decoded.shape}")

plt.tight_layout()
plt.show()
