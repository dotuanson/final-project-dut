import copy
import torch
from MODNet.src.models.modnet import MODNet
from MODNet.src.trainer import soc_adaptation_iter
import cv2
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

bs = 1  # batch size
lr = 0.00001  # learn rate
epochs = 10  # total epochs
height, width = 32, 32

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

        image = cv2.imread(os.path.join(self.root_dir, img_file), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_crop = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        image_PIL = Image.fromarray(image_crop)
        image_tensor = torch_transforms(image_PIL)

        cuda = torch.device('cuda')
        image_tensor = image_tensor.to(cuda)

        return image_tensor


if __name__ == '__main__':
    modnet = torch.nn.DataParallel(MODNet()).cuda()
    modnet.load_state_dict(torch.load("test.pt"))  # NOTE: please finish this function

    optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
    ImageMatting = ImageMattingDataset(txt_file="datasets/training.txt", root_dir="datasets")
    dataloader = DataLoader(ImageMatting, batch_size=bs, shuffle=True, num_workers=0)
    for epoch in range(0, epochs):
        backup_modnet = copy.deepcopy(modnet)
        for idx, (image) in enumerate(dataloader):
            soc_semantic_loss, soc_detail_loss = soc_adaptation_iter(modnet, backup_modnet, optimizer, image)
            print(soc_semantic_loss.data, soc_detail_loss.data)