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
import torchvision
import torchvision.transforms.v2 as transv2
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import non_max_suppression

from hackasaurus_rex.data import DroneImages
from hackasaurus_rex.detr import load_detr_model, postprocess_detr_ouput
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
    elif hyperparameters["model"] == "detr":
        return load_detr_model(hyperparameters["pretrained_weights"], freeze=True)
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


def load_model(hyperparameters, model, optimizer=None):
    if "model_checkpoint" in hyperparameters:
        checkpoint = torch.load(hyperparameters["checkpoint_path_in"])
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Restoring model checkpoint from {hyperparameters['checkpoint_path_in']}")
        return model


def get_bounding_box(prediction, mode):
    if mode == "yolo":
        postprocessed_prediction = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.7)
        return torch.cat([sample_prediction[:, 0:4] for sample_prediction in postprocessed_prediction])
        # boxes_for_metric = [{'boxes': sample_prediction[:, 0:4], 'masks': None}
        #                     for sample_prediction in postprocessed_prediction]
    elif mode == "detr":
        return postprocess_detr_ouput(prediction)
    else:
        raise NotImplementedError("What are you doing here!")


def train_epoch(
    model, optimizer, train_loader, train_metric, device, scaler, warmup_scheduler, lr_scheduler, hyperparameters
):
    # set the model into training mode
    model.train()
    rank = dist.get_rank() if dist.is_initialized() else 0

    # training procedure
    train_loss = 0.0
    metric_avg = 0.0
    avg_train_time = time.perf_counter()
    total_train_time = time.perf_counter()
    resize = transv2.Resize((2680, 3370))
    for i, batch in enumerate(train_loader):
        x, labels = batch
        x = torch.cat([i.unsqueeze(0) for i in x])
        x = x.to(device)
        labels = [{k: v.to(device) for k, v in label.items()} for label in labels]
        labels2 = []
        for lab in labels:
            labels2.append({k: v.to(device) for k, v in lab.items()})
            labels2[-1]["class_labels"] = lab["labels"]
        # for l in labels:
        #     print(l["boxes"].shape)
        # target_boxes = torch.cat([label["boxes"] for label in labels]).to(device)

        # print(target_boxes)
        model.zero_grad()

        with autocast(device_type="cuda", dtype=torch.float16, enabled=False):
            prediction = model(x, labels=labels2)
            # print(prediction)
            loss = prediction.loss
            # predicted_boxes = get_bounding_box(prediction, mode=hyperparameters["mode"])
            # print(predicted_boxes.shapes, target_boxes.shapes)
            # loss = torchvision.ops.generalized_box_iou_loss(predicted_boxes, target_boxes)
        # scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()
        with warmup_scheduler.dampening():
            pass
        train_loss += loss.item()

        # compute metric
        with torch.no_grad():
            # model.eval()
            # train_predictions = model(x)
            # metric needs:
            #   boxes -> list(dict("boxes"))
            #   masks ->
            target_shape = list(x.shape[:-2]) + [2680, 3370]
            metric_in = [
                {"boxes": prediction.pred_boxes[i], "masks": torch.empty(target_shape)}
                for i in range(prediction.pred_boxes.shape[0])
            ]
            _, metric_in = resize(x, metric_in)
            metric = train_metric(*to_mask(metric_in, labels)).item()
            # model.train()
        if rank == 0:  # and (i % 10 == 9 or i == len(train_loader) - 1):
            print(
                f"Train step {i}: metric: {metric:.4f} avg batch time: {(time.perf_counter() - avg_train_time) / (i + 1):.3f}"
            )
        metric_avg += metric
    if rank == 0:
        print(
            f"\nTrain epoch end: metric: {metric_avg / len(train_loader):.4f} total time: "
            f"{(time.perf_counter() - total_train_time)}s Memory utilized: {torch.cuda.max_memory_allocated()}\n"
        )

    return train_loss, train_metric.compute()


def evaluate(model, test_loader, test_metric, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    model.eval()
    iou_avg = 0
    resize = transv2.Resize((2680, 3370))
    for i, batch in enumerate(test_loader):
        x, labels = batch
        x = torch.cat([i.unsqueeze(0) for i in x])
        x = x.to(device)
        labels = [{k: v.to(device) for k, v in label.items()} for label in labels]
        labels2 = []
        for lab in labels:
            labels2.append({k: v.to(device) for k, v in lab.items()})
            labels2[-1]["class_labels"] = lab["labels"]
        # x_test = list(image.to(device) for image in x_test)
        # test_label = [{k: v.to(device) for k, v in label.items()} for label in test_label]

        # score_threshold = 0.7
        with torch.no_grad():
            prediction = model(x, labels=labels2)
            target_shape = list(x.shape[:-2]) + [2680, 3370]
            metric_in = [
                {"boxes": prediction.pred_boxes[i], "masks": torch.empty(target_shape)}
                for i in range(prediction.pred_boxes.shape[0])
            ]
            _, metric_in = resize(x, metric_in)
            iou = test_metric(*to_mask(metric_in, labels)).item()
            iou_avg += iou
            if rank == 0:
                print(f"Eval step: {i}: Avg iou: {iou_avg / (i + 1)}")

    end_iou = test_metric.compute()
    if rank == 0:
        print(f"End Eval: avg iou: {iou_avg / len(test_loader)} - {end_iou}\n")
    return end_iou


def train(hyperparameters):
    start_time = int(time.time())
    set_seed(hyperparameters["seed"])

    rank = dist.get_rank() if dist.is_initialized() else 0

    # determines the execution device, i.e. CPU or GPU
    device = get_device()
    print(f"Training on {device}")

    model = initialize_model(hyperparameters)
    model.to(device)
    if dist.is_initialized():
        model = DDP(model)  # , device_ids=[config.rank])

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
        num_workers=hyperparameters["data"]["workers"],
        pin_memory=hyperparameters["data"]["pin_memory"],
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
        pin_memory=hyperparameters["data"]["pin_memory"],
        sampler=test_sampler,
        collate_fn=collate_fn,
        persistent_workers=hyperparameters["data"]["persistent_workers"],
        prefetch_factor=hyperparameters["data"]["prefetch_factor"],
    )
    # End Dataloaders ---------------------------------------------------------------------

    # set up optimization procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hyperparameters["lr"]))  # , fused=True)
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

    scaler = GradScaler(enabled=False)

    # start the actual training procedure
    for epoch in range(hyperparameters["epochs"]):
        train_loss, train_metric_out = train_epoch(
            model,
            optimizer,
            train_loader,
            train_metric,
            device,
            scaler,
            warmup_scheduler,
            lr_scheduler,
            hyperparameters,
        )
        train_loss /= len(train_loader)

        test_metric_out = evaluate(model, test_loader, test_metric, device)

        if rank == 0:
            # output the losses
            print(f"Epoch {epoch}")
            print(f"\tTrain loss: {train_loss}")
            print(f"\tTrain IoU:  {train_metric_out}")
            print(f"\tTest IoU:   {test_metric_out}")

        # # save the best performing model on disk
        # if test_metric_out > best_iou and rank == 0:
        # best_iou = test_metric_out
        # print("\tSaving better model\n")
        # torch.save(model.state_dict(), "checkpoint.pt")
        # save_model(hyperparameters, model, optimizer, best_iou, start_time)
        elif rank == 0:
            print("\n")

        with warmup_scheduler.dampening():
            lr_scheduler.step()


def evaluation(hyperparameters):
    set_seed(hyperparameters["seed"])
    device = get_device()

    drone_images = DroneImages(hyperparameters["data"]["data_root"])
    if hyperparameters["split_data"]:
        _, test_data = torch.utils.data.random_split(drone_images, [0.8, 0.2], torch.Generator().manual_seed(42))
    else:
        test_data = drone_images
    test_data.train = False

    test_sampler = None
    if dist.is_initialized():
        test_sampler = datadist.DistributedSampler(test_data)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=hyperparameters["data"]["batch_size"],
        shuffle=False,
        num_workers=hyperparameters["data"]["workers"],
        pin_memory=hyperparameters["data"]["pin_memory"],
        sampler=test_sampler,
        collate_fn=collate_fn,
        persistent_workers=hyperparameters["data"]["persistent_workers"],
        prefetch_factor=hyperparameters["data"]["prefetch_factor"],
    )
    model = initialize_model(hyperparameters)
    model.to(device)
    if dist.is_initialized():
        model = DDP(model)
    load_model(hyperparameters, model)

    test_metric = IntersectionOverUnion(task="multiclass", num_classes=2)
    test_metric = test_metric.to(device)

    evaluate(model, test_loader, test_metric, device)

    print(f"Test IoU: {test_metric.compute()}")
