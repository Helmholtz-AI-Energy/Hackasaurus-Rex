import glob
import json
import os
import random
from multiprocessing import Process, Queue, set_start_method
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision
from torchvision import datapoints

torchvision.disable_beta_transforms_warning()
import torchvision
import torchvision.transforms.v2 as transv2
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DroneImages(torch.utils.data.Dataset):
    def __init__(self, root: str = "data", train=True):
        self.root = Path(root)
        self.parse_json(self.root / "descriptor.json")
        # TODO: add process for lazy staging to TMP
        self.tmp_dir = Path(os.environ["TMPDIR"])
        self.queue = Queue()
        self.check_staged = {name: False for name in self.ids}
        self.staging_proc = Process(target=self.stage, args=(self.queue, self.check_staged))
        self.staging_proc.daemon = True
        self.staging_proc.start()

        # print('testing that staging proc is alive', self.staging_proc.is_alive())

        _default_means = [130.0, 135.0, 135.0, 118.0, 118.0]
        _default_vars = [44.0, 40.0, 40.0, 30.0, 21.0]
        self.train = train
        self.first_trans = transv2.Compose(
            [
                transv2.ToImageTensor(),
                transv2.ConvertImageDtype(),
            ]
        )
        self.norm = transforms.Normalize(_default_means, _default_vars)

        self.transformations = transv2.Compose(
            [
                transv2.RandomHorizontalFlip(p=0.5),
                transv2.RandomVerticalFlip(p=0.5),
                # transv2.RandomRotation(90),
            ]
        )
        # self.first_trans = torch.jit.script(self.first_trans)
        self.grey = transforms.Grayscale()
        self.resize = transv2.Resize((1340, 1685))
        # self.transformations = torch.jit.script(transformations)

    @staticmethod
    def stage(queue, saved_dict):
        # need to change the value of self.images to load from temp
        while True:
            target, arr, name = queue.get()
            if not os.path.exists(target):
                np.save(target, arr)
                saved_dict[name] = True
            # if all(saved_dict):
            #     return

    def parse_json(self, path: Path):
        """
        Reads and indexes the descriptor.json

        The images and corresponding annotations are stored in COCO JSON format.
        This helper function reads out the images paths and segmentation masks.
        """
        with open(path, "r") as handle:
            content = json.load(handle)

        self.ids = [entry["id"] for entry in content["images"]]
        self.images = {entry["id"]: self.root / Path(entry["file_name"]).name for entry in content["images"]}

        # add all annotations into a list for each image
        self.polys = {}
        self.bboxes = {}
        for entry in content["annotations"]:
            image_id = entry["image_id"]
            self.polys.setdefault(image_id, []).append(entry["segmentation"])
            bbox = torch.tensor(entry["bbox"])
            bbox = torchvision.ops.box_convert(bbox, in_fmt="xywh", out_fmt="xyxy")
            self.bboxes.setdefault(image_id, []).append(bbox)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a drone image and its corresponding segmentation mask.

        The drone image is a tensor with dimensions [H x W x C=5], where
            H - height of the image
            W - width of the image
            C - (R,G,B,T,H) - five channels being red, green and blue color channels, thermal and depth information

        The corresponding segmentation mask is binary with dimensions [H x W].
        """
        image_id = self.ids[index]

        # deserialize the image from disk
        x = np.load(self.images[image_id])
        # if self.staging_proc.is_alive():
        save_loc = self.tmp_dir / f"{image_id}.npy"
        self.queue.put((save_loc, x, image_id))
        self.images[image_id] = save_loc

        x = torch.tensor(x, dtype=torch.float).permute((2, 0, 1))
        polys = self.polys[image_id]
        bboxes = self.bboxes[image_id]
        # masks = []
        # # generate the segmentation mask on the fly
        # for poly in polys:
        #     mask = Image.new(
        #         "L",
        #         (
        #             x.shape[1],
        #             x.shape[0],
        #         ),
        #         color=0,
        #     )
        #     draw = ImageDraw.Draw(mask)
        #     draw.polygon(poly[0], fill=1, outline=1)
        #     masks.append(np.array(mask))

        # masks = torch.tensor(np.array(masks))
        masks = torch.empty((0, x.shape[-2], x.shape[-1]))

        labels = torch.tensor([1 for a in polys], dtype=torch.int64)

        boxes = torch.tensor(bboxes, dtype=torch.float)
        boxes = datapoints.BoundingBox(boxes, spatial_size=(2680, 3370), format=datapoints.BoundingBoxFormat.XYWH)

        y = {
            "boxes": boxes,  # FloatTensor[N, 4]
            "labels": labels,  # Int64Tensor[N]
            "masks": masks.to(dtype=x.dtype, device=x.device),  # UIntTensor[N, H, W]
        }

        # x -> (R,G,B,T,H) x height x width
        x, y = self.first_trans(x, y)
        x = self.norm(x)
        grey = self.grey(x[:3])
        x = torch.cat([grey, x[3:]])
        x, y = self.resize(x, y)
        if self.train:
            x, y = self.transformations(x, y)
        return x, y


def create_train_val_split():
    # base location
    # /hkfs/work/workspace/scratch/ih5525-energy-train-data

    # set text file with paths to everything
    files = glob.glob("/hkfs/work/workspace/scratch/ih5525-energy-train-data/DJI*")
    num_files = len(files)
    random.shuffle(files)
    val_size = int(0.2 * num_files)
    train_list = files[:-val_size]
    val_list = files[-val_size:]
    with open("/hkfs/work/workspace/scratch/qv2382-hackathon/Hackasaurus-Rex/data/train.txt", "w") as f:
        for line in train_list:
            f.write(f"{line}\n")
    with open("/hkfs/work/workspace/scratch/qv2382-hackathon/Hackasaurus-Rex/data/val.txt", "w") as f:
        for line in val_list:
            f.write(f"{line}\n")


# if __name__ == '__main__':
#     create_train_val_split()
