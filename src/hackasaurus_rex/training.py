import random

import numpy as np
import torch
import torch.optim
import torch.utils.data
from tqdm import tqdm

from hackasaurus_rex.data import DroneImages
from hackasaurus_rex.metric import IntersectionOverUnion, to_mask


def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def initialize_model(hyperparameters):
    # TODO: create and initialize model, also load pretrained weights here
    pass


def load_model(hyperparameters):
    if "model_checkpoint" in hyperparameters:
        # TODO: initialize model and load checkpoint
        model = None
        print(f"Restoring model checkpoint from {hyperparameters['model_checkpoint']}")
        model.load_state_dict(torch.load(hyperparameters["model_checkpoint"]))
        return model
    else:
        raise ValueError("Please provide a model checkpoint.")


def train_epoch(model, optimizer, train_loader, train_metric, device):
    # set the model into training mode
    model.train()

    # training procedure
    train_loss = 0.0
    for i, batch in enumerate(tqdm(train_loader, desc="train")):
        x, labels = batch
        x = list(image.to(device) for image in x)
        labels = [{k: v.to(device) for k, v in label.items()} for label in labels]
        model.zero_grad()
        losses = model(x, labels)
        loss = sum(loss for loss in losses.values())

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # compute metric
        with torch.no_grad():
            model.eval()
            train_predictions = model(x)
            train_metric(*to_mask(train_predictions, labels))
            model.train()

    return train_loss


def evaluate(model, test_loader, test_metric, device):
    model.eval()

    for i, batch in enumerate(tqdm(test_loader, desc="test ")):
        x_test, test_label = batch
        x_test = list(image.to(device) for image in x_test)
        test_label = [{k: v.to(device) for k, v in label.items()} for label in test_label]

        # score_threshold = 0.7
        with torch.no_grad():
            test_predictions = model(x_test)
            test_metric(*to_mask(test_predictions, test_label))


def train(hyperparameters):
    set_seed(hyperparameters["seed"])

    # determines the execution device, i.e. CPU or GPU
    device = get_device()
    print(f"Training on {device}")

    # TODO: set up the dataset
    drone_images = DroneImages(hyperparameters["data_root"])
    train_data, test_data = torch.utils.data.random_split(drone_images, [0.8, 0.2])
    train_data.train = True
    test_data.train = False
    train_loader, test_loader = None, None

    # TODO: initialize the model
    model = initialize_model(hyperparameters)
    model.to(device)

    # set up optimization procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"])
    best_iou = 0.0

    train_metric = IntersectionOverUnion(task="multiclass", num_classes=2)
    train_metric = train_metric.to(device)
    test_metric = IntersectionOverUnion(task="multiclass", num_classes=2)
    test_metric = test_metric.to(device)

    # start the actual training procedure
    for epoch in range(hyperparameters["epochs"]):
        train_loss = train_epoch(model, optimizer, train_loader, train_metric, device)
        train_loss /= len(train_loader)

        evaluate(model, test_loader, test_metric, device)

        # output the losses
        print(f"Epoch {epoch}")
        print(f"\tTrain loss: {train_loss}")
        print(f"\tTrain IoU:  {train_metric.compute()}")
        print(f"\tTest IoU:   {test_metric.compute()}")

        # save the best performing model on disk
        if test_metric.compute() > best_iou:
            best_iou = test_metric.compute()
            print("\tSaving better model\n")
            torch.save(model.state_dict(), "checkpoint.pt")
        else:
            print("\n")


def eval(hyperparameters):
    set_seed(hyperparameters["seed"])
    device = get_device()

    drone_images = DroneImages(hyperparameters["data_root"])
    _, test_data = torch.utils.data.random_split(drone_images, [0.8, 0.2], torch.Generator().manual_seed(42))
    test_data.train = False

    test_loader = None
    model = load_model(hyperparameters)
    model.to(device)

    test_metric = IntersectionOverUnion(task="multiclass", num_classes=2)
    test_metric = test_metric.to(device)

    evaluate(model, test_loader, test_metric, device)

    print(f"Test IoU: {test_metric.compute()}")
