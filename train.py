import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
from baseline import tiramisu
from datasets import camvid
from datasets import joint_transforms
import utils.imgs
import utils.training as train_utils
from utils.training import save_result
from hyperparams import get_hyperparams

sys.setrecursionlimit(15000)
hyper = get_hyperparams()
dropout = hyper["dropout"]
mode = hyper["mode"]

assert mode in ["base", "epistemic", "aleatoric", "combined"], "Wrong mode!"

# CAMVID_PATH = Path('./CamVid/Camvid/')
CAMVID_PATH = "./CamVid/CamVid/"
# RESULTS_PATH = Path(".results/")
# WEIGHTS_PATH = Path(".weights/")

RESULTS_PATH = Path("/content/results/")
WEIGHTS_PATH = Path("/content/weights/")
SAVE_IMGS = Path("/content/combined/")

RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)
SAVE_IMGS.mkdir(exist_ok=True)

batch_size = hyper["batch_size"]

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
train_joint_transformer = transforms.Compose(
    [
        # joint_transforms.JointRandomCrop(224), # commented for fine-tuning
        joint_transforms.JointRandomHorizontalFlip()
    ]
)

# dataset prep
train_dset = camvid.CamVid(
    CAMVID_PATH,
    "train",
    # data augmentation for both image and label
    joint_transform=train_joint_transformer,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)

# dataloader creation
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=batch_size, shuffle=True
)

# dataset prep
val_dset = camvid.CamVid(
    CAMVID_PATH,
    "val",
    joint_transform=None,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)

# dataloader creation
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=False)

# dataset prep
test_dset = camvid.CamVid(
    CAMVID_PATH,
    "test",
    joint_transform=None,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)

# dataloader creation
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=batch_size, shuffle=False
)

print("Train: %d" % len(train_loader.dataset.imgs))
print("Val: %d" % len(val_loader.dataset.imgs))
print("Test: %d" % len(test_loader.dataset.imgs))
print("Classes: %d" % len(train_loader.dataset.classes))

inputs, targets = next(iter(train_loader))
print("Inputs: ", inputs.size())
print("Targets: ", targets.size())

utils.imgs.view_image(inputs[0])
utils.imgs.view_annotated(targets[0])

# _criterion = nn.NLLLoss2d(weight=camvid.class_weight.cuda(), reduction="none").cuda()
_criterion = nn.NLLLoss(weight=camvid.class_weight.cuda(), reduction="none").cuda()


def custom_cirterion(y_pred, y_true):
    """Aleatoric loss function
    See paper at 2.2 Heteroscedastic Aleatoric Uncertainty (5)
    """
    output, log_var = y_pred
    log_var = log_var[0]
    loss = torch.exp(-1 * log_var) * 0.5 + _criterion(output, y_true) + 0.5 * log_var
    return loss.sum()

def custom_cirterion(y_pred, y_true):
  
    T=50 # number of mc samples for stochastic approximation

    logits, log_var = y_pred
    batch_size, num_classes, height, width = logits.size()

    # print("logits shape:", logits.shape)
    # print("log_var shape:", log_var.shape)

    # reshape to match predictions
    y_true = y_true.view(batch_size, -1)  
    logits = logits.view(batch_size, num_classes, -1)  
    log_var = log_var.view(batch_size, 1, -1) 

    # equation 12 in the paper 
    # sample random number from normal dist
    epsilon = torch.randn((T, batch_size, 1, logits.size(-1)), device=logits.device)
    # std_dev = exp(log_var/2) [convert log_var to std dev]
    std_dev = torch.exp(0.5 * log_var).unsqueeze(0)
    
    # sample from the logits
    print("Logits shape:", logits.shape)
    print("Log variance shape:", log_var.shape)
    print("Target shape:", y_true.shape)

    perturbed_logits = logits.unsqueeze(0) + std_dev * epsilon
    # perturbed_logits = logits.unsqueeze(0) + log_var * epsilon
    softmax_outputs = nn.functional.softmax(perturbed_logits, dim=2)

    print("Min probability:", prob_ave.min().item())
    print("Max probability:", prob_ave.max().item())

    print("Target min:", y_true.min().item())
    print("Target max:", y_true.max().item())

    prob_ave = torch.mean(softmax_outputs, 0)

    total_loss = _criterion(torch.log(prob_ave), y_true)
    total_loss = total_loss.sum()
    # total_loss = total_loss.sum() / y_true.numel()
    return total_loss

# iou
def iou_calculation(pred, target, n_classes=12):
    pred = pred.flatten()
    target = target.flatten()
    ious = []


    for cls in range(n_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        if union > 0:
            ious.append(intersection / union)

    return np.mean(ious)

# precision recall
def precision_recall(pred, target, n_classes=12):

    precision = []
    recall = []
    for cls in range(n_classes):
        tp = ((pred == cls) & (target == cls)).sum()
        fp = ((pred == cls) & (target != cls)).sum()
        fn = ((pred != cls) & (target == cls)).sum()

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)

        precision.append(prec)
        recall.append(rec)
    return np.mean(precision), np.mean(recall)

# entropy
def compute_entropy_map(prob_map):
    prob_map = prob_map.cpu().numpy()
    entropy_map = -np.sum(prob_map * np.log(prob_map + 1e-12), axis=0)
    return entropy_map

# hyperparameters
LR = hyper["learning_rate"]
LR_DECAY = hyper["lr_decay"]
DECAY_EVERY_N_EPOCHS = hyper["decay_per_n_epoch"]
N_EPOCHS = hyper["n_epoch"]

torch.cuda.manual_seed(0)

if mode == "base":
    model = tiramisu.FCDenseNet57(n_classes=12, dropout=dropout).cuda()
    model.apply(train_utils.weights_init)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.NLLLoss(weight=camvid.class_weight.cuda()).cuda()
    test = train_utils.test
    train = train_utils.train

elif mode == "epistemic":
    model = tiramisu.FCDenseNet57(n_classes=12, dropout=dropout).cuda()
    model.apply(train_utils.weights_init)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.NLLLoss(weight=camvid.class_weight.cuda()).cuda()
    test = train_utils.test_epistemic
    train = train_utils.train

elif mode == "aleatoric":
    model = tiramisu.FCDenseNet57_aleatoric(n_classes=12, dropout=dropout).cuda()
    model.apply(train_utils.weights_init)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = custom_cirterion
    test = train_utils.test_aleatoric
    train = train_utils.train_aleatoric

elif mode == "combined":
    model = tiramisu.FCDenseNet57_aleatoric(n_classes=12, dropout=dropout).cuda()
    model.apply(train_utils.weights_init)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = custom_cirterion
    test = train_utils.test_combined
    train = train_utils.train_aleatoric

if __name__ == "__main__":
    # Initialize metric tracking
    iou_scores = []
    precisions = []
    recalls = []
    entropies = []

    # Training loop
    val_tmp = sys.maxsize
    print("Mode: {}".format(mode))
    for epoch in range(1, N_EPOCHS + 1):
        since = time.time()
        
        # Train
        trn_loss, trn_err = train(model, train_loader, optimizer, criterion, epoch)
        print(
            "Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}".format(
                epoch, trn_loss, 1 - trn_err
            )
        )
        
        time_elapsed = time.time() - since
        print("Train Time {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        # Test
        val_loss, val_err = test(model, val_loader, criterion, epoch)
        print("Val - Loss: {:.4f} | Acc: {:.4f}".format(val_loss, 1 - val_err))
        
        time_elapsed = time.time() - since
        print("Total Time {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))

        # if val_tmp < val_loss:  # early stopping
        #     break
        # else:
        #     val_tmp = val_loss
        save_result(trn_loss, trn_err, val_loss, val_err, epoch)

        with torch.no_grad():
            iou_epoch = []
            precision_epoch = []
            recall_epoch = []
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                
                # Obtain predictions
                # preds = torch.argmax(outputs[0], dim=1) if mode != "base" else torch.argmax(outputs, dim=1)
                preds = torch.argmax(outputs[0], dim=1)
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()

                # iou, precision, recall
                iou_epoch.append(iou_calculation(preds_np, targets_np))
                precision, recall = precision_recall(preds_np, targets_np)
                precision_epoch.append(precision)
                recall_epoch.append(recall)

                if mode in ["aleatoric", "combined"]:       #entropy
                    prob_map = torch.nn.functional.softmax(outputs[0], dim=1)
                    entropy_map = compute_entropy_map(prob_map)
                    entropies.append(entropy_map.mean())

            iou_scores.append(np.mean(iou_epoch))
            precisions.append(np.mean(precision_epoch))
            recalls.append(np.mean(recall_epoch))

            print(f"IoU: {iou_scores[-1]:.4f}, Precision: {precisions[-1]:.4f}, Recall: {recalls[-1]:.4f}")

        train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS)

    # Save final model weights
    # weights_filename = WEIGHTS_PATH / f"model_epoch_{epoch}_val_loss_{val_loss:.4f}.pth"
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'val_loss': val_loss,
    #     'val_err': val_err,
    # }, weights_filename)
    # print(f"Model weights saved to {weights_filename}")

    weights_filename = WEIGHTS_PATH / f"model_epoch_{epoch}_val_loss_{val_loss:.4f}.pth"
    torch.save({
        # 'epoch': epoch,
        'startEpoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'val_loss': val_loss,
        # 'val_err': val_err,
        'train_loss': trn_loss,
        'train_err': trn_err,
        'loss': val_loss,
        'error': val_err,

    }, weights_filename)
    print(f"Model weights saved to {weights_filename}")

    # Plot metrics after training
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(iou_scores) + 1), iou_scores, label="IoU", marker="o")
    plt.plot(range(1, len(precisions) + 1), precisions, label="Precision", marker="o")
    plt.plot(range(1, len(recalls) + 1), recalls, label="Recall", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("IoU, Precision, and Recall over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("/content/combined/test_iou", bbox_inches="tight") 
    plt.show()
    time.sleep(5) 

    # Plot entropy map if applicable
    if mode in ["aleatoric", "combined"]:
        plt.figure()
        plt.hist(entropies, bins=30, alpha=0.7, label="Entropy")
        plt.xlabel("Entropy")
        plt.ylabel("Frequency")
        plt.title("Entropy Distribution")
        plt.legend()
        plt.savefig("/content/combined/test_entropy", bbox_inches="tight") 
        plt.show()
        time.sleep(5) 

    # Plot Precision vs Recall
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, label="Precision vs Recall", marker="o", linestyle="-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall")
    plt.legend()
    plt.grid()
    plt.savefig("/content/combined/precision_vs_recall", bbox_inches="tight") 
    plt.show()
    time.sleep(5)