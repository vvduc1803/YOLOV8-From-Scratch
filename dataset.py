from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
import json
from metrics import iou_width_height as iou

class MyDataset(Dataset):
    def __init__(self, img_dir, json_file, strides=[8, 16, 32], reg_max=16, num_classes=80, image_size=640, transform=None, threshold=0.5):
        self.json_dir = json_file
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.reg_max = reg_max
        self.img_size = image_size
        self.transform = transform
        self.strides = strides
        self.num_classes = num_classes
        self.threshold = threshold


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # Take picture
        image_path = os.path.join(self.img_dir, self.img_list[idx])
        image = np.array(Image.open(image_path))


        # Extract label and bbox of objects in file csv
        json_dir = open(self.json_dir, 'r')
        json_file = json.load(json_dir)
        images_info, annotations = json_file['images'], json_file['annotations']
        image_info = [[image['id'], image['width'], image['height']]
                    for image in images_info if image['file_name'] == f'{self.img_list[idx]}'][0]

        bboxes = [sum([anno['bbox'], [anno['category_id']]], [])
                  for anno in annotations
                  if anno['image_id'] == image_info[0]]

        self.class_idx = [i['id'] for i in json_file['categories']]

        # Transform image and bounding boxes
        if self.transform:
            augmentation = self.transform(image=image, bboxes=bboxes)
            image = augmentation['image']
            bboxes = augmentation['bboxes']

        targets = self.convert_labels_to_target(bboxes)

        return image, targets

    def convert_labels_to_target(self, bboxes):
        targets = [torch.zeros((grid_size, grid_size, self.num_classes + 5)) for grid_size in (8, 16, 32)]
        for box in bboxes:
            x, y, w, h, class_label = box
            xc, yc = x + w / 2, y + h / 2

            for i, grid_size in enumerate((8, 16, 32)):
                if grid_size == 8:
                    anchor_idxs = [0, 1, 2]
                elif grid_size == 16:
                    anchor_idxs = [3, 4, 5]
                else:
                    anchor_idxs = [6, 7, 8]

                stride = self.image_size // grid_size
                x_cell, y_cell = int(xc / stride), int(yc / stride)
                grid_x, grid_y = xc / stride - x_cell, yc / stride - y_cell

                for idx in anchor_idxs:
                    if targets[i][x_cell, y_cell, 20] == 0:
                        targets[i][x_cell, y_cell, 20] = 1
                        targets[i][x_cell, y_cell, 21:25] = torch.tensor([x, y, w, h])
                        targets[i][x_cell, y_cell, int(class_label)] = 1

        return image, tuple(targets)



a = MyDataset('coco2017/val2017', 'coco2017/annotations/instances_val2017.json')
x, b = a.__getitem__(0)
print(len(b))
