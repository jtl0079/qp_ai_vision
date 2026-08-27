from torch.utils.data import Dataset
import cv2
import torchvision
import torch
##extend Dataset class from torch to let the class can traning
class SsdTorchDataset(Dataset):

    def __init__(self, ssd_dataset,transform=None,split="train"):
        self.ssd_dataset = ssd_dataset
        self.transform = transform

        self.image_paths = []

        for image_path, annotation in ssd_dataset.annotations.items():

            if split == "train":
                if "/train/" in image_path.replace("\\", "/"):
                    self.image_paths.append(image_path)

            elif split == "valid":
                if "/valid/" in image_path.replace("\\", "/"):
                    self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        annotation = self.ssd_dataset.annotations[
            image_path
        ]

        target = {
            "boxes": torch.tensor([
            [
                obj.bbox.x_min,
                obj.bbox.y_min,
                obj.bbox.x_max,
                obj.bbox.y_max,
            ]
            for obj in annotation.objects],dtype=torch.float32,)
            ,
            "labels": torch.tensor(
                [obj.class_id+1 for obj in annotation.objects],
                dtype=torch.int64,
            ),
        }


        if self.transform is not None:
          image,target = self.transform(image,target)
        else:
          image = torchvision.transforms.ToTensor()(image)
        
        


        return image, target