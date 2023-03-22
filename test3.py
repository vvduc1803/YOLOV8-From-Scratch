import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class AnchorFreeDataset(Dataset):
    def __init__(self, path_to_images, path_to_labels, num_classes, input_sizes, transform=None):
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.images_list = os.listdir(self.path_to_images)
        self.labels_list = os.listdir(self.path_to_labels)
        self.num_classes = num_classes
        self.input_sizes = input_sizes
        self.transform = transform

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        # Extract label and bbox of objects in file csv
        label_path = os.path.join(self.path_to_labels, self.labels_list[idx])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, 1).tolist()

        # Take picture
        image_path = os.path.join(self.path_to_images, self.images_list[idx])
        image = np.array(Image.open(image_path))

        # Transform image and bounding boxes
        if self.transform:
            augmentation = self.transform(image=image, bboxes=bboxes)
            image = augmentation['image']
            bboxes = augmentation['bboxes']

        # Create empty target tensors for each scale
        targets = []
        for input_size in self.input_sizes:
            target = torch.zeros((self.num_classes + 5, input_size, input_size))
            targets.append(target)

        # Parse the label information
        for bbox in bboxes:

            x_center = bbox[0]
            y_center = bbox[1]
            width = bbox[2]
            height = bbox[3]
            class_label = int(bbox[4])
            # Calculate the grid cell coordinates for each scale
            grid_x = [int(x_center * input_size) for input_size in self.input_sizes]
            grid_y = [int(y_center * input_size) for input_size in self.input_sizes]
            print(grid_x)
            print(grid_y)

            # Calculate the normalized coordinates for the largest scale
            x_center /= self.input_sizes[-1]
            y_center /= self.input_sizes[-1]

            # Set the confidence score to 1 for each scale

            for i in range(len(self.input_sizes)):
                targets[i][self.num_classes, grid_y[i], grid_x[i]] = 1

            # Set the bounding box coordinates for each scale
            for i, input_size in enumerate(self.input_sizes):

                targets[i][self.num_classes + 1, grid_y[i], grid_x[i]] = x_center - grid_x[i] * (input_size // 32)
                targets[i][self.num_classes + 2, grid_y[i], grid_x[i]] = y_center - grid_y[i] * (input_size // 32)
                targets[i][self.num_classes + 3, grid_y[i], grid_x[i]] = width / input_size
                targets[i][self.num_classes + 4, grid_y[i], grid_x[i]] = height / input_size

            # Set the class label for each scale
            for i in range(len(self.input_sizes)):
                targets[i][class_label, grid_y[i], grid_x[i]] = 1

        return image, tuple(targets)

a = AnchorFreeDataset('image', 'labels', 20, [20, 40, 80])
img , tup = a.__getitem__(0)
print(tup[2][:, 28, 38])