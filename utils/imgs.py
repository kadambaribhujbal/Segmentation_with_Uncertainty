import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Define label colors
Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

DSET_MEAN = [0.41189489566336, 0.4251328133025, 0.4326707089857]
DSET_STD = [0.27413549931506, 0.28506257482912, 0.28284674400252]

label_colours = np.array(
    [
        Sky,
        Building,
        Pole,
        Road,
        Pavement,
        Tree,
        SignSymbol,
        Fence,
        Car,
        Pedestrian,
        Bicyclist,
        Unlabelled,
    ]
)

# Save directory for images
SAVE_IMGS = Path("/content/combined/")
SAVE_IMGS.mkdir(exist_ok=True)


def view_annotated(tensor, plot=True, n=0, path=SAVE_IMGS, mode="target"):
    """
    Visualize and optionally save the annotated image based on segmentation labels.
    """
    temp = tensor.numpy()
    r, g, b = temp.copy(), temp.copy(), temp.copy()
    
    for l in range(0, len(label_colours)):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    path = "/content/combined/"

    save_path = path + "{}-{}.png".format(n, mode)

    if plot:
        plt.imshow(rgb)
        plt.title(mode)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        # plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        return rgb


def decode_image(tensor):
    """
    Decode an image tensor into a visualizable format.
    """
    inp = tensor.numpy().transpose((1, 2, 0))
    mean = np.array(DSET_MEAN)
    std = np.array(DSET_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # Clip to valid range
    return inp


def view_image(tensor, plot=True, path=SAVE_IMGS, n=0, mode="pred"):
    """
    Visualize and optionally save an image.
    """
    path = "/content/combined/"
    inp = decode_image(tensor)
    save_path = path + "{}-{}.png".format(n, mode)

    if plot:
        plt.imshow(inp)
        plt.title(mode)
        plt.savefig(save_path, bbox_inches="tight")  # Save plot
        plt.show()
        # plt.savefig(save_path, bbox_inches="tight")  # Save plot
        plt.close()
    else:
        return inp


def view_image_with_uncertainty(tensor1, tensor2, path=SAVE_IMGS, n=0, mode="epistemic"):
    """
    Visualize and optionally save an image with uncertainty overlay.
    """
    inp = decode_image(tensor1)
    inp = np.clip(inp, 0, 1)
    tensor2 = tensor2.to(torch.device("cpu")).detach().numpy()[0]

    path = "/content/combined/"

    save_path = path + "{}-{}.png".format(n, mode)

    plt.imshow(inp)
    plt.pcolor(tensor2, alpha=0.5, cmap='viridis')  # Uncertainty overlay
    plt.colorbar(label='Uncertainty')
    plt.title(mode)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()
