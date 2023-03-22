from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
from metrics import iou_width_height as iou

class MyDataset(Dataset):
    def __init__(self, img_dir, label_dir, S, anchors, num_classes = 20, image_size=416, transform=None, threshold=0.5):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.label_list = os.listdir(label_dir)
        self.image_size = image_size
        self.transform = transform
        self.input_sizes = S
        self.num_classes = num_classes
        # self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        # self.num_anchors = self.anchors.shape[0]
        # self.number_anchors_per_scale = self.num_anchors // 3
        self.threshold = threshold


    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        # Extract label and bbox of objects in file csv
        label_path = os.path.join(self.label_dir, self.label_list[idx])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, 1).tolist()

        # Take picture
        image_path = os.path.join(self.img_dir, self.img_list[idx])
        image = np.array(Image.open(image_path))

        # Transform image and bounding boxes
        if self.transform:
            augmentation = self.transform(image=image, bboxes=bboxes)
            image = augmentation['image']
            bboxes = augmentation['bboxes']

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
            # Create empty target tensors for each scale
        targets = []
        for input_size in self.input_sizes:
            target = torch.zeros((self.num_classes + 5, input_size, input_size))
            targets.append(target)
        
        for bbox in bboxes:
            iou_anchors = iou(torch.tensor(bbox[2:4]), self.anchors)
            anchors_indices = torch.argsort(iou_anchors, dim=0, descending=True)
            has_anchors = [False]*3
            x, y, width, height, class_label = bbox

            for anchors_index in anchors_indices:
                scale_index = anchors_index // self.number_anchors_per_scale
                anchor_on_scale = scale_index % self.number_anchors_per_scale
                S = self.S[scale_index]
                i, j = int(S * x), int(S * y)
                anchor_taken = targets[scale_index][anchor_on_scale, j, i, 0]
                if not anchor_taken and not has_anchors[scale_index]:
                    targets[scale_index][anchor_on_scale, j, i, 0] = 1

                    x_cell, y_cell, width_cell, height_cell = S*x - i, S*y -j, width*S, height*S
                    box = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_index][anchor_on_scale, j, i, 1:5] = box
                    targets[scale_index][anchor_on_scale, j, i, 5] = class_label

                    has_anchors[scale_index] = True

                if not anchor_taken and iou_anchors[scale_index] > self.threshold:
                    targets[scale_index][anchor_on_scale, j, i, 0] = -1

        return image, tuple(targets)

a = MyDataset('image', 'labels', [32, 16, 8], None)
a.__getitem__(0)
