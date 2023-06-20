import random
import time
from pathlib import Path

import numpy as np
import pytorch_warmup as warmup
import torch
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed as datadist
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from ultralytics import YOLO

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
    if hyperparameters["model"] == "yolo":
        return load_yolo_model(hyperparameters["pretrained_weights"], freeze=True)
    else:
        raise NotImplementedError(f'Model {hyperparameters["model"]} not supported.')


def load_yolo_model(pretrained_weights, freeze, num_classes=1):
    print(f"Loading YOLO model from {pretrained_weights}")
    model = YOLO(pretrained_weights).model
    unfreeze(model)

    # adjust detection heads for new number of classes and reset the weights
    if num_classes != model.nc:
        detection_heads = model.model[22]
        for class_prediction_head in detection_heads.cv3:
            class_prediction_head[0].conv.out_channels = num_classes
            class_prediction_head[0].bn.num_features = num_classes
            class_prediction_head[0].conv.reset_parameters()
            class_prediction_head[0].bn.reset_parameters()

            class_prediction_head[1].conv.in_channels = num_classes
            class_prediction_head[1].conv.out_channels = num_classes
            class_prediction_head[1].bn.num_features = num_classes
            class_prediction_head[1].conv.reset_parameters()
            class_prediction_head[1].bn.reset_parameters()

            class_prediction_head[2].in_channels = num_classes
            class_prediction_head[2].out_channels = num_classes
            class_prediction_head[2].reset_parameters()

    # freeze all but first layer and heads
    if freeze:
        # model.0 is the first layer, model.22 are the object detection heads
        parameters_to_freeze = [
            parameter
            for parameter_name, parameter in model.named_parameters()
            if not (parameter_name.startswith("model.0") or parameter_name.startswith("model.22"))
        ]

        for param in parameters_to_freeze:
            param.requires_grad = False

    return model


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def save_model(hyperparameters, model, optimizer, best_iou, start_time):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    fname = f"{start_time}_{best_iou.item()}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        Path(hyperparameters["checkpoint_path_out"]) / fname,
    )


def load_model(hyperparameters, model, optimizer):
    if "model_checkpoint" in hyperparameters:
        checkpoint = torch.load(hyperparameters["checkpoint_path_in"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Restoring model checkpoint from {hyperparameters['checkpoint_path_in']}")
        return model


def train_epoch(model, optimizer, train_loader, train_metric, device, scaler, warmup_scheduler, lr_scheduler):
    # set the model into training mode
    model.train()
    rank = dist.get_rank() if dist.is_initialized() else 0

    # training procedure
    train_loss = 0.0
    metric_avg = 0.0
    avg_train_time = time.perf_counter()
    total_train_time = time.perf_counter()
    for i, batch in enumerate(train_loader):
        x, labels = batch
        x = list(image.to(device) for image in x)
        labels = [{k: v.to(device) for k, v in label.items()} for label in labels]
        model.zero_grad()

        with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            losses = model(x, labels)
            # TODO: FIXME?
            loss = sum(loss for loss in losses.values())
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()
        with warmup_scheduler.dampening():
            pass
        train_loss += loss.item()

        # compute metric
        with torch.no_grad():
            model.eval()
            train_predictions = model(x)
            metric = train_metric(*to_mask(train_predictions, labels)).item()
            model.train()
        if rank == 0 and (i % 10 == 9 or i == len(train_loader) - 1):
            print(
                f"Train step {i}: metric: {metric:.4f} avg batch time: {(time.perf_counter() - avg_train_time) / i:.3f}"
            )
        metric_avg += metric
    if rank == 0:
        print(
            f"\nTrain epoch end: metric: {metric_avg / len(train_loader):.4f} total time: "
            f"{(time.perf_counter() - total_train_time)}s Memory utilized: {torch.cuda.max_memory_allocated()}\n"
        )

    return train_loss


def evaluate(model, test_loader, test_metric, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    model.eval()
    iou_avg = 0
    resize_to_large = transforms.v2.Resize((2680, 3370))
    for i, batch in enumerate(test_loader):
        x_test, test_label = batch
        x_test, test_label = resize_to_large(x_test, test_label)
        # x_test = list(image.to(device) for image in x_test)
        # test_label = [{k: v.to(device) for k, v in label.items()} for label in test_label]

        # score_threshold = 0.7
        with torch.no_grad():
            test_predictions = model(x_test)
            iou = test_metric(*to_mask(test_predictions, test_label))
            iou_avg += iou
            if rank == 0 and (i % 10 == 9 or i == len(test_loader) - 1):
                print(f"Eval step: {i}: Avg iou: {iou_avg / i}")
    if rank == 0:
        print(f"End Eval: avg iou: {iou / len(test_loader)}\n")


def train(hyperparameters):
    start_time = int(time.time())
    set_seed(hyperparameters["seed"])

    rank = dist.get_rank() if dist.is_initialized() else 0

    # determines the execution device, i.e. CPU or GPU
    device = get_device()
    print(f"Training on {device}")

    model = initialize_model(hyperparameters)
    if dist.is_initialized():
        model = DDP(model)  # , device_ids=[config.rank])
    model.to(device)

    # TODO: set up the dataset
    drone_images = DroneImages(hyperparameters["data"]["data_root"])
    train_data, test_data = torch.utils.data.random_split(drone_images, [0.8, 0.2])
    train_data.train = True
    test_data.train = False

    # Dataloaders -------------------------------------------------------------------------
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_sampler = None
    shuffle = True
    if dist.is_initialized():
        train_sampler = datadist.DistributedSampler(train_data)
        shuffle = False

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=hyperparameters["data"]["batch_size"],
        shuffle=shuffle,
        num_workers=6,
        pin_memory=hyperparameters["data"]["workers"],
        sampler=train_sampler,
        persistent_workers=hyperparameters["data"]["persistent_workers"],
        collate_fn=collate_fn,
        prefetch_factor=hyperparameters["data"]["prefetch_factor"],
    )

    test_sampler = None
    shuffle = False
    if dist.is_initialized():
        test_sampler = datadist.DistributedSampler(test_data)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=hyperparameters["data"]["batch_size"],
        shuffle=shuffle,
        num_workers=hyperparameters["data"]["workers"],
        pin_memory=True,
        sampler=test_sampler,
        persistent_workers=hyperparameters["data"]["persistent_workers"],
        prefetch_factor=hyperparameters["data"]["prefetch_factor"],
    )
    # End Dataloaders ---------------------------------------------------------------------

    # set up optimization procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hyperparameters["lr"]), fused=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        400,
        gamma=0.1,
    )
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=200)

    load_model(hyperparameters, model, optimizer)

    best_iou = 0.0

    train_metric = IntersectionOverUnion(task="multiclass", num_classes=2)
    train_metric = train_metric.to(device)
    test_metric = IntersectionOverUnion(task="multiclass", num_classes=2)
    test_metric = test_metric.to(device)

    scaler = GradScaler(enabled=True)

    # start the actual training procedure
    for epoch in range(hyperparameters["epochs"]):
        train_loss = train_epoch(model, optimizer, train_loader, train_metric, device, scaler)
        train_loss /= len(train_loader)

        evaluate(model, test_loader, test_metric, device)

        if rank == 0:
            # output the losses
            print(f"Epoch {epoch}")
            print(f"\tTrain loss: {train_loss}")
            print(f"\tTrain IoU:  {train_metric.compute()}")
            print(f"\tTest IoU:   {test_metric.compute()}")

        # save the best performing model on disk
        if test_metric.compute() > best_iou and rank == 0:
            best_iou = test_metric.compute()
            print("\tSaving better model\n")
            # torch.save(model.state_dict(), "checkpoint.pt")
            save_model(hyperparameters, model, optimizer, best_iou, start_time)
        elif rank == 0:
            print("\n")

        with warmup_scheduler.dampening():
            lr_scheduler.step()


def evaluation(hyperparameters):
    set_seed(hyperparameters["seed"])
    device = get_device()

    drone_images = DroneImages(hyperparameters["data"]["data_root"])
    _, test_data = torch.utils.data.random_split(drone_images, [0.8, 0.2], torch.Generator().manual_seed(42))
    test_data.train = False

    test_sampler = None
    if dist.is_initialized():
        test_sampler = datadist.DistributedSampler(test_data)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=hyperparameters["data"]["batch_size"],
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        sampler=test_sampler,
        persistent_workers=hyperparameters["data"]["persistent_workers"],
        prefetch_factor=hyperparameters["data"]["prefetch_factor"],
    )
    model = initialize_model(hyperparameters)
    # TODO: load our model checkpoint
    model.to(device)

    test_metric = IntersectionOverUnion(task="multiclass", num_classes=2)
    test_metric = test_metric.to(device)

    evaluate(model, test_loader, test_metric, device)

    print(f"Test IoU: {test_metric.compute()}")
