import torch
import os
from MODNet.src.models.modnet import MODNet
from MODNet.src.trainer import supervised_training_iter, soc_adaptation_iter
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2
from scipy.ndimage import grey_dilation, grey_erosion
import numpy as np
from torch.utils.data import DataLoader

bs = 2  # batch size
lr = 0.01  # learning rate
epochs = 40  # total epochs
height, width = 512, 672

modnet = torch.nn.DataParallel(MODNet()).cuda()
optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class ImageMattingDataset(Dataset):
    """Image Matting dataset."""

    def __init__(self, txt_file, root_dir, transform=None):
        with open(txt_file, "r") as file:
            self.txt_file = file.readlines()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.txt_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_file = self.txt_file[idx].split('\t')[0]
        matte_file = self.txt_file[idx].split('\t')[1].strip()

        image = cv2.imread(os.path.join(self.root_dir, img_file), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        matte = cv2.imread(os.path.join(self.root_dir, matte_file), 0)

        image_crop = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        matte_crop = cv2.resize(matte, (width, height), interpolation=cv2.INTER_NEAREST)

        image_PIL = Image.fromarray(image_crop)
        image_tensor = torch_transforms(image_PIL)

        matte_PIL = Image.fromarray(matte_crop)
        matte_tensor = transforms.ToTensor()(matte_PIL)

        h, w = matte_crop.shape[:2]
        side = int((h + w) / 2 * 0.05)
        fg = matte_crop > 25
        dilated = grey_dilation(fg, size=(side, side))
        eroded = grey_erosion(fg, size=(side, side))
        unknown = dilated & (1 - eroded)
        unknown = np.uint8(unknown) * 128
        trimap_crop = np.where(unknown == 128, 128, matte_crop)
        trimap_PIL = Image.fromarray(trimap_crop)
        trimap_tensor = transforms.ToTensor()(trimap_PIL)
        condition = (trimap_tensor > 0) & (trimap_tensor < 1)
        trimap_tensor = torch.where(condition, torch.scalar_tensor(0.5, dtype=torch.float32), trimap_tensor)

        cuda = torch.device('cuda')
        image_tensor = image_tensor.to(cuda)
        matte_tensor = matte_tensor.to(cuda)
        trimap_tensor = trimap_tensor.to(cuda)

        return image_tensor, trimap_tensor, matte_tensor


if __name__ == '__main__':
    image_matting = ImageMattingDataset(txt_file="datasets/training.txt", root_dir="datasets")
    dataloader = DataLoader(image_matting, batch_size=bs, shuffle=True, num_workers=0)
    for epoch in range(0, epochs):
        for idx, (image, trimap, gt_matte) in enumerate(dataloader):
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(modnet, optimizer, image, trimap,
                                                                              gt_matte)
            print(semantic_loss.data, detail_loss.data, matte_loss.data)
        lr_scheduler.step()
