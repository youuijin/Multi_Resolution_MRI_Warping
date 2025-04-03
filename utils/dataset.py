import torch, os
import nibabel as nib
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

def set_dataloader(image_paths, template_path, batch_size):
    dataset = MedicalImageDataset(image_paths, template_path)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)  # 80%
    val_size = int(dataset_size * 0.1)    # 10%
    test_size = dataset_size - train_size - val_size  # remain

    # split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    save_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, save_loader


# Define dataset class
class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, template_path, transform=None):
        self.image_paths = []
        self.image_paths = [f'{root_dir}/{f}' for f in os.listdir(root_dir)]
        self.template = nib.load(template_path).get_fdata()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx])
        affine = img.affine
        img = img.get_fdata()
        img_min, img_max = img.min(), img.max()

        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]
        template = (self.template - self.template.min()) / (self.template.max() - self.template.min())
        
        if self.transform:
            img = self.transform(img)
        
        return torch.tensor(img, dtype=torch.float32), torch.tensor(template, dtype=torch.float32), img_min, img_max, affine
