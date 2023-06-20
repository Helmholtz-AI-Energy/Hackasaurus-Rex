import glob
import json
import os
import random
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Tuple

import h5py as h5
import numpy as np
import torch
import torchvision
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
        self.staging_proc.start()
        _default_means = [130.0, 135.0, 135.0, 118.0, 118.0]
        _default_vars = [44.0, 40.0, 40.0, 30.0, 21.0]
        self.train = train
        self.first_trans = torch.nn.Sequential(
            transforms.v2.ToTensor(),
            transforms.Normalize(_default_means, _default_vars),
        )

        transformations = torch.nn.Sequential(
            transforms.v2.RandomHorizontalFlip(p=0.5),
            transforms.v2.RandomVerticalFlip(p=0.5),
            # transforms.v2.RandomRotation(90),
        )
        self.first_trans = torch.jit.script(self.first_trans)
        self.grey = transforms.Greyscale()
        self.resize = transforms.v2.Resize((1340, 1685))
        self.transformations = torch.jit.script(transformations)

    @staticmethod
    def stage(queue, saved_dict):
        # need to change the value of self.images to load from temp
        while True:
            target, arr, name = queue.get()
            if not os.path.exists(target):
                arr.save(target)
                saved_dict[name] = True
            if all(saved_dict):
                return

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
            self.bboxes.setdefault(image_id, []).append(entry["bbox"])

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
        if self.staging_proc.is_alive():
            save_loc = self.tmp_dir / image_id
            self.queue.put((save_loc, x, image_id))
            self.images[image_id] = save_loc

        polys = self.polys[image_id]
        bboxes = self.bboxes[image_id]
        masks = []
        # generate the segmentation mask on the fly
        for poly in polys:
            mask = Image.new(
                "L",
                (
                    x.shape[1],
                    x.shape[0],
                ),
                color=0,
            )
            draw = ImageDraw.Draw(mask)
            draw.polygon(poly[0], fill=1, outline=1)
            masks.append(np.array(mask))

        masks = torch.tensor(np.array(masks))

        labels = torch.tensor([1 for a in polys], dtype=torch.int64)

        boxes = torch.tensor(bboxes, dtype=torch.float)
        # bounding boxes are given as [x, y, w, h] but rcnn expects [x1, y1, x2, y2]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        y = {
            "boxes": boxes,  # FloatTensor[N, 4]
            "labels": labels,  # Int64Tensor[N]
            "masks": masks,  # UIntTensor[N, H, W]
        }
        x = torch.tensor(x, dtype=torch.float).permute((2, 0, 1))
        # x -> (R,G,B,T,H) x height x width
        x, y = self.first_trans(x, y)
        grey = self.grey(x[:3])
        x = torch.cat([grey, x[3:]])
        x = self.resize(x)
        if self.train:
            self.transformations(x, y)
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
