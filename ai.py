#
# Credit: https://www.kaggle.com/code/danielhavir/vesuvius-challenge-example-submission
#
############################################## Imports:
import os
import gc
from pathlib import Path
from typing import List, Tuple
import warnings
import util
import shutil
import opendatasets as od
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import PIL.Image as Image
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thd
from tqdm import tqdm

from config import Config

################################################## Data:
class SubvolumeDataset(thd.Dataset):
    def __init__(
        self,
        fragments: List[Path],
        voxel_shape: Tuple[int, int, int],
        load_inklabels: bool = True,
        filter_edge_pixels: bool = False,
    ):
        self.fragments = sorted(map(lambda path: path.resolve(), fragments))
        self.voxel_shape = voxel_shape
        self.load_inklabels = load_inklabels
        self.filter_edge_pixels = filter_edge_pixels

        # Load sequentially
        labels = []
        image_stacks = []
        valid_pixels = []
        for fragment_id, fragment_path in enumerate(self.fragments):
            fragment_path = fragment_path.resolve()  # absolute path
            mask = np.array(Image.open(Path.joinpath(fragment_path, "mask.png")).convert("1"))

            surface_volume_paths = sorted(
                (fragment_path / "surface_volume").rglob("*.tif")
            )
            z_dim, y_dim, x_dim = voxel_shape

            z_mid = len(surface_volume_paths) // 2
            z_start, z_end = z_mid - z_dim // 2, z_mid + z_dim // 2

            # we don't convert to torch since it doesn't support uint16
            images = [
                np.array(Image.open(fn)) for fn in surface_volume_paths[z_start:z_end]
            ]
            image_stack = np.stack(images, axis=0)
            image_stacks.append(image_stack)

            pixels = np.stack(np.where(mask == 1), axis=1).astype(np.uint16)
            if filter_edge_pixels:
                height, width = mask.shape
                mask_y = np.logical_or(
                    pixels[:, 0] < y_dim // 2, pixels[:, 0] >= height - y_dim // 2
                )
                mask_x = np.logical_or(
                    pixels[:, 1] < x_dim // 2, pixels[:, 1] >= width - x_dim // 2
                )
                pixel_mask = np.logical_or(mask_y, mask_x)
                pixels = pixels[~pixel_mask]
            # encode fragment ID
            fragment_ids = np.full_like(pixels[:, 0:1], fragment_id)
            pixels = np.concatenate((pixels, fragment_ids), axis=1)
            valid_pixels.append(pixels)

            if load_inklabels:
                # binary mask can be stored as np.bool
                inklabels = (
                    np.array(Image.open(Path.joinpath(fragment_path, "inklabels.png"))) > 0
                )
                labels.append(inklabels)

            print(f"Loaded fragment {fragment_path} on {os.getpid()}")

        self.labels = labels
        self.image_stacks = image_stacks
        self.pixels = np.concatenate(valid_pixels).reshape(
            -1, valid_pixels[0].shape[-1]
        )

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        center_y, center_x, fragment_id = self.pixels[index]
        z_dim, y_dim, x_dim = self.voxel_shape
        image_stack = self.image_stacks[fragment_id]
        _, height, width = image_stack.shape

        # pad with zeros if necessary
        if (
            center_y < y_dim // 2
            or center_x < x_dim // 2
            or center_y + y_dim // 2 >= height
            or center_x + x_dim // 2 >= width
        ):
            # calculate the upper-left corner of the sub-volume
            y_start = max(center_y - y_dim // 2, 0)
            x_start = max(center_x - x_dim // 2, 0)

            # calculate the lower-right corner of the sub-volume
            y_end = min(center_y + y_dim // 2, height)
            x_end = min(center_x + x_dim // 2, width)

            subvolume = np.zeros(self.voxel_shape, dtype=np.float32)

            pad_y_start = max(y_dim // 2 - center_y, 0)
            pad_x_start = max(x_dim // 2 - center_x, 0)

            pad_y_end = min(height + y_dim // 2 - center_y, y_dim)
            pad_x_end = min(width + x_dim // 2 - center_x, x_dim)

            subvolume[:, pad_y_start:pad_y_end, pad_x_start:pad_x_end] = (
                image_stack[:, y_start:y_end, x_start:x_end].astype(np.float32) / 65535
            )

        else:
            subvolume = (
                image_stack[
                    :,
                    center_y - y_dim // 2 : center_y + y_dim // 2,
                    center_x - x_dim // 2 : center_x + x_dim // 2,
                ]
            ).astype(np.float32) / 65535
        if self.load_inklabels:
            inklabel = float(self.labels[fragment_id][center_y, center_x])
        else:
            inklabel = -1.0

        return torch.from_numpy(subvolume).unsqueeze(0), torch.FloatTensor([inklabel])

    def plot_label(self, index, **kwargs):
        pixel = self.pixels[index]
        label = self.labels[pixel[-1]]

        print("Index:", index)
        print("Pixel:", pixel)
        print("Label:", int(label[pixel[0], pixel[1]]))

        if isinstance(label, torch.Tensor):
            label = label.numpy()

        fig, ax = plt.subplots(**kwargs)
        ax.imshow(label, cmap="gray")

        y, x, _ = pixel
        _, y_dim, x_dim = self.voxel_shape
        x_min = x - (x_dim // 2)
        x_max = x + (x_dim // 2)
        y_min = y - (y_dim // 2)
        y_max = y + (y_dim // 2)
        rect = Rectangle(
            (x_min, y_min), x_dim, y_dim, linewidth=2, edgecolor="y", facecolor="none"
        )
        ax.add_patch(rect)
        plt.show()
        plt.show()


class InkDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        filters = [16, 32, 64]
        paddings = [1, 1, 1]
        kernel_sizes = [3, 3, 3]
        strides = [2, 2, 2]
        
        layers = []
        in_channels = 1
        for num_filters, padding, kernel_size, stride in zip(filters, paddings, kernel_sizes, strides):
            layers.extend([
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.ReLU(inplace=True),
                torch.nn.BatchNorm3d(num_features=num_filters)
            ])
            in_channels = num_filters
        layers.append(nn.AdaptiveAvgPool3d(1))
        layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

# This is the basic model architecture for the InkDetector.

"""
################################################## Better InkDetector:
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm3d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class InkDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder using Residual Blocks
        self.encoder = nn.Sequential(
            ResidualBlock(1, 16, stride=2),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool3d(1),  # Output shape: (batch_size, 128, 1, 1, 1)
            nn.Flatten()              # Output shape: (batch_size, 128)
        )
        # Decoder: MLP with Dropout
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)  # Output is a single logit (use sigmoid externally if needed)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # This will be a single logit; use `torch.sigmoid(x)` if you need probability
"""

################################################## AI:
class AI:
    def __init__(self, batch_size: int, training_steps: int, learning_rate: float):
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        config = Config()
        if config.get("trained") == "true":
            self.train_run = False
        else:
            self.train_run = True
        
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InkDetector().to(self.DEVICE)
        
        self.base_path = ""
    
    def set_basepath(self, path: str):
        self.base_path = path

    def train_model(self, train_loader, config: Config):
        print("Model Generated, training model...")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=self.training_steps)
        self.model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        running_fbeta = 0.0
        denom = 0
        pbar = tqdm(enumerate(train_loader), total=self.training_steps)
        for i, (subvolumes, inklabels) in pbar:
            if i >= self.training_steps:
                break
            optimizer.zero_grad()
            outputs = self.model(subvolumes.to(self.DEVICE))
            loss = criterion(outputs, inklabels.to(self.DEVICE))
            loss.backward()
            optimizer.step()
            scheduler.step()
            pred_ink = outputs.detach().sigmoid().gt(0.4).cpu().int()
            accuracy = (pred_ink == inklabels).sum().float().div(inklabels.size(0))
            running_fbeta += fbeta_score(inklabels.view(-1).numpy(), pred_ink.view(-1).numpy(), beta=0.5)
            running_accuracy += accuracy.item()
            running_loss += loss.item()
            denom += 1
            pbar.set_postfix({"Loss": running_loss / denom, "Accuracy": running_accuracy / denom, "Fbeta@0.5": running_fbeta / denom})
            if (i + 1) % 500 == 0:
                running_loss = 0.
                running_accuracy = 0.
                running_fbeta = 0.
                denom = 0

        torch.save(self.model.state_dict(), os.path.join(self.base_path, "data/model.pt"))
        config.edit("trained", "true")
        config.save()

    def load_model(self, training_data: list, config: Config):
        train_path = os.path.join(self.base_path, "data/vesuvius-challenge-ink-detection/train")
        
        print("Train Run: " + str(self.train_run))
        if self.train_run:
            all_fragments = sorted(training_data)
            print("All fragments to train with:", all_fragments)

            # load amount of fragments for training:
            train_fragments = [Path(os.path.join(train_path, fragment_name)) for fragment_name in training_data]
            train_dset = SubvolumeDataset(fragments=train_fragments, voxel_shape=(48, 64, 64), filter_edge_pixels=True)
            train_loader = thd.DataLoader(train_dset, batch_size=self.batch_size, shuffle=True)

            print("Num batches:", len(train_loader))
            print("Num items (pixels)", len(train_dset))
            print("Loaded dataset for training, generating model...")
            
            warnings.simplefilter('ignore', UndefinedMetricWarning)
            
            self.train_model(train_loader, config)
            
            # Clear memory to free RAM:
            train_dset.labels = []
            train_dset.image_stacks = []
            del train_loader, train_dset
            gc.collect()
        else:
            print("Loading model configurations into memory...")
            model_weights = torch.load(os.path.join(self.base_path, "data/model.pt"))
            self.model.load_state_dict(model_weights)

        print("Model successfully loaded.")

    def eval_model(self, threshold: float):
        # Test:
        test_path = os.path.join(self.base_path, "data/vesuvius-challenge-ink-detection/test")
        test_fragments = [test_path / fragment_name for fragment_name in Path(test_path).iterdir()]
        print("All fragments to run: ", test_fragments)
        pred_images = []
        self.model.eval()
        for test_fragment in test_fragments:
            outputs = []
            eval_dset = SubvolumeDataset(fragments=[test_fragment], voxel_shape=(48, 64, 64), load_inklabels=False)
            eval_loader = thd.DataLoader(eval_dset, batch_size=self.batch_size, shuffle=False)
            with torch.no_grad():
                for i, (subvolumes, _) in enumerate(tqdm(eval_loader)):
                    output = self.model(subvolumes.to(self.DEVICE)).view(-1).sigmoid().cpu().numpy()
                    outputs.append(output)
            # only load 1 fragment at a time
            image_shape = eval_dset.image_stacks[0].shape[1:]
            eval_dset.labels = []
            eval_dset.image_stacks = []
            del eval_loader
            gc.collect()

            pred_image = np.zeros(image_shape, dtype=np.uint8)
            outputs = np.concatenate(outputs)
            for (y, x, _), prob in zip(eval_dset.pixels[:outputs.shape[0]], outputs):
                pred_image[y, x] = prob > threshold 
            pred_images.append(pred_image)
            
            eval_dset.pixels = np.empty((0, 3), dtype=np.uint16)
            del eval_dset
            gc.collect()
            print("Finished this segment-> ", test_fragment)
        
        util.clear()
        print("Finished! saving ink images...")
            
        config = Config()
        
        for i, pred_image in enumerate(pred_images): # todo
            plt.imshow(pred_image, cmap='gray')
            plt.axis('off')
            plt.gca().set_position([0, 0, 1, 1])
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            
            file_name = f"image_{threshold}_{i}.png"
            if not os.path.isdir("outputs"):
                os.mkdir(config.get("base_path") + "/outputs")
            
            plt.savefig(os.path.join(config.get("base_path") + "/outputs/", file_name), format='png', bbox_inches='tight', pad_inches=0)
            print(f"Saved {file_name}")

def download_data():    
    try:
        print("Starting Download...")
        od.download("https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data", "data/")
        print("Download Complete!")
        
    except KeyboardInterrupt:
        print("ERROR: exiting prematurly. Data installation incomplete.")
        print("deleting corrupt data...")
        if os.path.isdir("data"):
            shutil.rmtree("data")
        exit(1)
