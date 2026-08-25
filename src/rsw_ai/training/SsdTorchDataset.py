from torch.utils.data import Dataset
import cv2
import torchvision
import torch

class SsdTorchDataset(Dataset):

    def __init__(self, ssd_dataset,transform=None,):
        self.ssd_dataset = ssd_dataset
        self.image_paths = list(ssd_dataset.annotations.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        if self.transform is not None:
          image = self.transform(image)
        else:
          image = torchvision.transforms.ToTensor()(image)
        
        annotation = self.ssd_dataset.annotations[
            image_path
        ]

        target = {
            "boxes": torch.tensor(
                annotation.boxes,
                dtype=torch.float32,
            ),
            "labels": torch.tensor(
                annotation.labels,
                dtype=torch.int64,
            ),
        }


        return image, target