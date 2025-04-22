import torch, os
import nibabel as nib
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import numpy as np

def set_dataloader(image_paths, template_path, batch_size, feature_load=False, pre_loss=None):
    dataset = MedicalImageDataset(image_paths, template_path, feature_load=feature_load, pre_loss=pre_loss)
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
    def __init__(self, root_dir, template_path, feature_load=False, pre_loss=None, transform=None):
        self.image_paths = []
        self.image_paths = [f'{root_dir}/{f}' for f in os.listdir(root_dir)]
        template = nib.load(template_path).get_fdata()
        self.template = (template - template.min()) / (template.max() - template.min())
        self.transform = transform
        self.feature_load = feature_load
        if feature_load:
            if pre_loss == "MSE":
                self.pre_path = 'data/brain_U-Net_MSE'
            elif pre_loss == "NCC":
                self.pre_path = 'data/brain_U-Net_NCC'
            else:
                raise ValueError('Pre-loss type error', pre_loss)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_name = self.image_paths[idx].split('/')[-1]

        img = np.load(f"{self.image_paths[idx]}/data.npy")
        affine = np.load(f"{self.image_paths[idx]}/affine.npy")
        
        img_min, img_max = img.min(), img.max()

        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]
        
        if self.transform:
            img = self.transform(img)

        if self.feature_load:
            disps = [np.load(f"{self.pre_path}/{image_name}/disp4.npy"), np.load(f"{self.pre_path}/{image_name}/disp2.npy"), np.load(f"{self.pre_path}/{image_name}/disp1.npy")]
            features = [np.load(f"{self.pre_path}/{image_name}/feat4.npy"), np.load(f"{self.pre_path}/{image_name}/feat2.npy"), np.load(f"{self.pre_path}/{image_name}/feat1.npy")]
            return torch.tensor(img, dtype=torch.float32), torch.tensor(self.template, dtype=torch.float32), img_min, img_max, affine, disps, features
        
        return self.image_paths[idx], torch.tensor(img, dtype=torch.float32), torch.tensor(self.template, dtype=torch.float32), img_min, img_max, affine
    
    # # previous version: nii load
    # def __getitem__(self, idx):
    #     img = nib.load(self.image_paths[idx])
    #     affine = img.affine
    #     img = img.get_fdata()
    #     img_min, img_max = img.min(), img.max()

    #     img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]
    #     template = (self.template - self.template.min()) / (self.template.max() - self.template.min())
        
    #     if self.transform:
    #         img = self.transform(img)
        
    #     return torch.tensor(img, dtype=torch.float32), torch.tensor(template, dtype=torch.float32), img_min, img_max, affine
