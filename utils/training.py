import os
import sys
import math
import string
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from hyperparams import get_hyperparams
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from . import imgs as img_utils

# RESULTS_PATH = ".results/"
# WEIGHTS_PATH = ".weights/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_PATH = "/content/results/"
WEIGHTS_PATH = "/content/weights/"


hyper = get_hyperparams()
batch_size = hyper["batch_size"]
num_classes = hyper["num_classes"]
img_shape = hyper["image_shape"]
mode = hyper["mode"]
lr = hyper["learning_rate"]


def save_weights(model, epoch, loss, err, mode):
    weights_fname = "weights-%s-%d-%.3f-%.3f.pth" % (mode, epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save(
        {
            "startEpoch": epoch,
            "loss": loss,
            "error": err,
            # "val_loss": loss,
            # "val_err": err,
            "state_dict": model.state_dict(),
        },
        weights_fpath,
    )
    shutil.copyfile(weights_fpath, WEIGHTS_PATH + "latest.th")


# def load_weights(model, fpath):
#     print("loading weights '{}'".format(fpath))

#     weights = torch.load(fpath, weights_only=True)
#     # weights = torch.load(fpath)
#     startEpoch = weights["startEpoch"]
#     # model.load_state_dict(weights["state_dict"])
#     model.load_state_dict(weights["model_state_dict"])
#     print(
#         "loaded weights (lastEpoch {}, loss {}, error {})".format(
#             startEpoch - 1, weights["loss"], weights["error"]
#         )
#     )
#     return startEpoch

def load_weights(model, fpath):
    print(f"Loading weights from '{fpath}'")

    # Determine the appropriate device for loading
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the weights and map them to the correct device
    weights = torch.load(fpath, map_location=map_location)

    startEpoch = weights["startEpoch"]
    
    # Load the model state dict
    model.load_state_dict(weights["model_state_dict"])

    print(
        f"Loaded weights (lastEpoch {startEpoch - 1}, loss {weights['loss']}, error {weights['error']})"
    )

    return startEpoch


# output the predicted class index
def get_predictions(output_batch):
    bs, c, h, w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs, h, w)
    return indices


def get_epistemic(outputs, predictive_mean, test_trials=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    result = torch.tensor(
        np.zeros((batch_size, img_shape[0], img_shape[1]), dtype=np.float32)
    ).to(device)
    target_sq = torch.einsum("bchw,bchw->bhw", [predictive_mean, predictive_mean]).data
    for i in range(test_trials):
        output_sq = torch.einsum(
            "bchw,bchw->bhw", [outputs[i], outputs[i]]
        ).data 
        result += output_sq - target_sq
    result /= test_trials
    return result


def error(preds, targets):
    assert preds.size() == targets.size()
    bs, h, w = preds.size()
    n_pixels = bs * h * w
    incorrect = float(preds.ne(targets).cpu().sum())
    err = incorrect / n_pixels
    return round(err, 5)


def train(model, trn_loader, optimizer, criterion, epoch):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    trn_loss = 0
    trn_error = 0
    for idx, data in enumerate(trn_loader):
        if list(data[0].size())[0] != batch_size:
            break
        # inputs = Variable(data[0].cuda())
        # targets = Variable(data[1].cuda())

        inputs = data[0].to(device)
        targets = data[1].to(device)

        optimizer.zero_grad()
        output = model(inputs)[0]  #only logits
        loss = criterion(output, targets)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        # trn_loss += loss.item()
        trn_loss += loss.data

        pred = get_predictions(output)
        trn_error += error(pred, targets.data.cpu())
        # trn_error += error(pred, targets.cpu())

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    torch.cuda.empty_cache()
    return float(trn_loss), trn_error


def train_aleatoric(model, trn_loader, optimizer, criterion, epoch):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Train aleatoric model"""
    model.train()
    trn_loss = 0
    trn_error = 0
    for idx, data in enumerate(trn_loader):
        if list(data[0].size())[0] != batch_size:
            break
        # inputs = Variable(data[0].cuda())
        # targets = Variable(data[1].cuda())

        inputs = data[0].to(device)
        targets = data[1].to(device)


        optimizer.zero_grad()
        output, logvar = model(inputs)  # tuple of [logits, log_var]
        loss = criterion((output, logvar), targets)
        loss.backward()
        optimizer.step()
        # trn_loss += loss.item()
        trn_loss += loss.data
        pred = get_predictions(output)
        trn_error += error(pred, targets.data.cpu())
        # trn_error += error(pred, targets.cpu())
    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    torch.cuda.empty_cache()
    return float(trn_loss), trn_error


def test(model, test_loader, criterion, epoch=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Baseline Test
    model: pytorch model
    test_loader: test data loader
    crieterion: loss_fucntion
    Return
        test_loss, test_error
    """
    model.eval()  # dropout off
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        if list(data.size())[0] != batch_size:
            break
        # data = Variable(data.cuda(), volatile=True)
        # target = Variable(target.cuda())

        with torch.no_grad():
            data = data.to(device)

        target = target.to(device)

        output = model(data)[0]
        # test_loss += criterion(output, target).item()
        test_loss += criterion(output, target).data
        pred = get_predictions(output)
        test_error += error(pred, target.data.cpu())
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    torch.cuda.empty_cache()
    return test_loss, test_error


def test_aleatoric(model, test_loader, criterion, epoch=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Baseline Test
    model: pytorch model
    test_loader: test data loader
    crieterion: loss_fucntion
    Return
        test_loss, test_error
    """
    model.eval()  # dropout off
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        if list(data.size())[0] != batch_size:
            break
        # data = Variable(data.cuda(), volatile=True)
        # target = Variable(target.cuda())

        with torch.no_grad():
            data = data.to(device)

        target = target.to(device)

        output = model(data)
        # test_loss += criterion(output, target).item()
        test_loss += criterion(output, target).data

        pred = get_predictions(output[0])
        test_error += error(pred, target.data.cpu())
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    torch.cuda.empty_cache()
    return test_loss, test_error


def test_epistemic(model, test_loader, criterion, test_trials=20, epoch=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Epistemic model Test
    Please turn on Dropout!
    model: pytorch model
    test_loader: test data loader
    crieterion: loss_fucntion
    Return
        test_loss, test_error
    """
    model.train()  # train mode: turn on dropout but weights are not updated
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        # print(torch.cuda.memory_allocated(device=0))
        if list(data.size())[0] != batch_size:
            break
        # data = Variable(data.cuda(), volatile=True)
        # target = Variable(target.cuda())

        with torch.no_grad():
            data = data.to(device)

        target = target.to(device)


        outputs = model(data)[0].data
        for i in range(test_trials - 1):
            # outputs += model(data)[0].data
            outputs += model(data)[0].data
        output = outputs / test_trials  # mean
        pred = get_predictions(output)
        test_loss += criterion(output, target).data
        # test_loss += criterion(output, target).item()
        test_error += error(pred, target.data.cpu())
        torch.cuda.empty_cache()
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    torch.cuda.empty_cache()
    return test_loss, test_error


def test_combined(model, test_loader, criterion, test_trials=20, epoch=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Combined model Test
    Please turn on Dropout!
    model: pytorch model
    test_loader: test data loader
    crieterion: loss_fucntion
    Return
        test_loss, test_error
    """
    model.train()  # train mode: turn on dropout
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        if list(data.size())[0] != batch_size:
            break
        # data = Variable(data.cuda(), volatile=True)
        with torch.no_grad():
            data = data.to(device)

        # target = Variable(target.cuda())

        target = target.to(device)


        outputs, log_var = model(data)
        outputs = outputs.data
        for i in range(test_trials - 1):
            outputs += model(data)[0].data
            # outputs += model(data)[0].detach()
        output = outputs / test_trials
        pred = get_predictions(output)
        test_loss += criterion((output, log_var), target).data
        test_error += error(pred, target.data.cpu())

        # test_loss += criterion((output, log_var), target).item()
        # test_error += error(pred, target.cpu())

        torch.cuda.empty_cache()
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    torch.cuda.empty_cache()
    return test_loss, test_error


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         # nn.init.kaiming_uniform(m.weight)
#         nn.init.kaiming_uniform_(m.weight)
#         m.bias.data.zero_()

# gaussian prior
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()


def predict(model, input_loader, n_batches=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:

        # data = Variable(input.cuda(), volatile=True)
        # label = Variable(target.cuda())
        with torch.no_grad():
            data = data.to(device)
        label = target.to(device)
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input, target, pred])
    return predictions


def view_sample_predictions(model, loader, n):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, targets = next(iter(loader))
    # data = Variable(inputs.cuda(), volatile=True)
    # label = Variable(targets.cuda())
    with torch.no_grad():
            data = inputs.to(device)
    label = targets.to(device)
    
    output = model(data)[0]
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        img_utils.view_annotated(targets[i])
        img_utils.view_annotated(pred[i])


# def view_sample_predictions_with_uncertainty(
#     model, inputs, targets, n, test_trials=2
# ):
#     # Dropout for MC Sampling
#     model.train()
#     # data = Variable(inputs.cuda(), volatile=True).view(1, 3, img_shape[0], img_shape[1])
#     with torch.no_grad():
#         data = inputs.cuda().view(1, 3, img_shape[0], img_shape[1])
#     label = targets.cuda()
#     # label = Variable(targets.cuda())
#     output, log_var = model(data)
#     shape = (1, 1, num_classes, img_shape[0], img_shape[1])
#     outputs = model(data)[0].view(shape).data
#     for i in range(test_trials - 1):
#         output = model(data)[0].view(shape).data
#         outputs = torch.cat([outputs, output], dim=0)
#     predictive_mean = outputs.mean(dim=0)  # mean
#     pred = get_predictions(predictive_mean)[0]
#     base_path = "./combined/"
#     # base_path = "/content/combined/"
#     # uncertainty
#     epistemic = get_epistemic(outputs, predictive_mean, test_trials)  # check shape
#     aleatoric = log_var[0]

#     img_utils.view_image(inputs, path=base_path, n=n, mode="input")
#     img_utils.view_annotated(targets, path=base_path, n=n, mode="target")
#     img_utils.view_annotated(pred, path=base_path, n=n, mode="pred")
#     img_utils.view_image_with_uncertainty(
#         inputs, epistemic, path=base_path, n=n, mode="epistemic"
#     )
#     img_utils.view_image_with_uncertainty(
#         inputs, aleatoric, path=base_path, n=n, mode="aleatoric"
#     )


def view_sample_predictions_with_uncertainty(
    model, inputs, targets, sample_id, test_trials=100
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    Visualizes predictions with uncertainty (epistemic and aleatoric) for a given model and input.

    Args:
        model (torch.nn.Module): The trained model.
        inputs (torch.Tensor): Input image tensor.
        targets (torch.Tensor): Ground truth labels.
        sample_id (int): Unique identifier for the sample being visualized.
        test_trials (int): Number of Monte Carlo trials for uncertainty estimation.

    Returns:
        None
    """
    # Set model to training mode for Monte Carlo (MC) Dropout
    model.train()
    
    # Prepare input tensor and move to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device).unsqueeze(0)  # Add batch dimension
    targets = targets.to(device)
    
    # Run initial forward pass
    with torch.no_grad():
        output, log_var = model(inputs)  # `log_var` represents aleatoric uncertainty
    
    # Collect predictions from multiple MC trials
    outputs = []
    for _ in range(test_trials):
        trial_output, _ = model(inputs)
        outputs.append(trial_output.unsqueeze(0))
    
    # Stack outputs and compute predictive mean
    outputs = torch.cat(outputs, dim=0)  # Shape: [test_trials, 1, num_classes, H, W]
    predictive_mean = outputs.mean(dim=0).squeeze(0)  # Shape: [num_classes, H, W]
    
    # Add batch dimension for compatibility with `get_predictions`
    predictive_mean = predictive_mean.unsqueeze(0)  # Shape: [1, num_classes, H, W]
    
    # Generate final prediction and uncertainties
    predictions = get_predictions(predictive_mean)  # Now `predictive_mean` is 4D
    epistemic_uncertainty = get_epistemic(outputs, predictive_mean, test_trials)
    aleatoric_uncertainty = log_var.squeeze(0).cpu()  # Shape: [H, W]
    
    # Define base path for saving visualizations
    base_path = "./combined/"
    os.makedirs(base_path, exist_ok=True)
    
    # Visualization
    img_utils.view_image(inputs[0].cpu(), path=base_path, n=sample_id, mode="input")
    img_utils.view_annotated(targets.cpu(), path=base_path, n=sample_id, mode="target")
    img_utils.view_annotated(predictions[0], path=base_path, n=sample_id, mode="pred")
    img_utils.view_image_with_uncertainty(
        inputs[0].cpu(), epistemic_uncertainty.cpu(), path=base_path, n=sample_id, mode="epistemic"
    )
    img_utils.view_image_with_uncertainty(
        inputs[0].cpu(), aleatoric_uncertainty, path=base_path, n=sample_id, mode="aleatoric"
    )




# def save_result(
#     train_loss: float, train_err: float, val_loss: float, val_error: float, epoch: int
# ) -> None:

    # save = hyper
    # save["train_loss"] = train_loss
    # save["train_err"] = train_err
    # save["loss"] = val_loss
    # save["error"] = val_error
    # save_ = sorted(save.items(), key=(lambda x: x[0]))
    # dataframe = pd.DataFrame(save_)
    # dataframe.to_csv(
    #     "/content/results/{}-lr-{}-epoch-{}.csv".format(mode, lr, epoch), encoding="utf-8"
    # )

def save_result(epoch, train_loss, train_err, val_loss, val_err, iou, precision, recall, entropy) -> None:
    """
    Save the metrics for a given epoch.
    """

    save = hyper
    save["train_loss"] = train_loss
    save["train_err"] = train_err
    save["val_loss"] = val_loss
    save["val_err"] = val_err
    save["iou"] = iou
    save["precision"] = precision
    save["recall"] = recall
    save["entropy"] = entropy
    save_ = sorted(save.items(), key=(lambda x: x[0]))
    dataframe = pd.DataFrame(save_)
    dataframe.to_csv(
        "/content/results/{}-lr-{}-epoch-{}.csv".format(mode, lr, epoch), encoding="utf-8"
    )
    # global results

    # results["epoch"].append(epoch)
    # results["train_loss"].append(train_loss)
    # results["train_err"].append(train_err)
    # results["val_loss"].append(val_loss)
    # results["val_err"].append(val_err)
    # results["iou"].append(iou)
    # results["precision"].append(precision)
    # results["recall"].append(recall)
    # results["entropy"].append(entropy)

    # # Save the DataFrame to CSV after each epoch
    # df = pd.DataFrame(results)
    # df.to_csv("/content/results/{}-lr-{}-metrics.csv".format(mode, lr), index=False)
