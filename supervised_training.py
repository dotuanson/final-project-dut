import torch
import os
import cv2
import numpy as np
import yaml
from tqdm import tqdm
from scipy.ndimage import grey_dilation, grey_erosion
from PIL import Image

from MODNet.src.models.modnet import MODNet
from MODNet.src.trainer import supervised_training_iter
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as transforms
from torch.multiprocessing import set_start_method
from torchvision.utils import make_grid

with open("config.yaml") as f:
    config = yaml.safe_load(f)

writer = SummaryWriter(os.path.join("runs", config["training"]["summary_writer"]))

modnet = torch.nn.DataParallel(MODNet()).cuda()
optimizer = torch.optim.SGD(modnet.parameters(), lr=config["hyper_parameter"]["learning_rate"], momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["hyper_parameter"]["lr_schedule_epoch"],
                                               gamma=0.1)

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

        image_crop = cv2.resize(image,
                                (config["hyper_parameter"]["size"]["width"],
                                 config["hyper_parameter"]["size"]["height"]),
                                interpolation=cv2.INTER_NEAREST)
        matte_crop = cv2.resize(matte,
                                (config["hyper_parameter"]["size"]["width"],
                                 config["hyper_parameter"]["size"]["height"]),
                                interpolation=cv2.INTER_NEAREST)

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
    # Load weight cua tac gia
    if config["training"]["weight_pretrained"] is not None:
        weights = torch.load(config["training"]["weight_pretrained"])
        modnet.load_state_dict(weights)

    train_image_matting = ImageMattingDataset(txt_file=config["training"]["training_file"],
                                              root_dir=config["training"]["dataset_root"])
    train_dataloader = DataLoader(train_image_matting,
                                  batch_size=config["hyper_parameter"]["batch_size"],
                                  num_workers=config["training"]["num_workers"],
                                  shuffle=True,
                                  drop_last=False)

    valid_image_matting = ImageMattingDataset(txt_file=config["training"]["validation_file"],
                                              root_dir=config["training"]["dataset_root"])
    valid_dataloader = DataLoader(valid_image_matting,
                                  batch_size=config["hyper_parameter"]["batch_size"],
                                  num_workers=config["training"]["num_workers"],
                                  shuffle=False,
                                  drop_last=False)

    image, _, _ = next(iter(train_dataloader))
    writer.add_graph(modnet, image)

    best_mse = 2**31-1
    best_mad = 2**31-1

    for epoch in tqdm(range(0, config["hyper_parameter"]["epochs"])):
        # training
        sum_semantic_loss = 0.0
        sum_detail_loss = 0.0
        sum_matte_loss = 0.0
        n_sample = 0
        for image, trimap, gt_matte in train_dataloader:
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(modnet, optimizer, image, trimap,
                                                                              gt_matte)
            sum_semantic_loss += semantic_loss.item() * image.shape[0]
            sum_detail_loss += detail_loss.item() * image.shape[0]
            sum_matte_loss += matte_loss.item() * image.shape[0]
            n_sample += image.shape[0]
        avg_semantic_loss = sum_semantic_loss / n_sample
        avg_detail_loss = sum_detail_loss / n_sample
        avg_matte_loss = sum_matte_loss / n_sample
        writer.add_scalar("Train/Semantic loss", avg_semantic_loss, epoch)
        writer.add_scalar("Train/Detail loss", avg_detail_loss, epoch)
        writer.add_scalar("Train/Matte loss", avg_matte_loss, epoch)
        lr_scheduler.step()

        # validation
        if (epoch + 1) % config["training"]["print_validation"] == 0:
            with torch.no_grad():
                modnet.eval()
                mse = 0.0
                mad = 0.0
                n_sample = 0
                for image, trimap, gt_matte in valid_dataloader:
                    _, _, pred_matte = modnet(image)
                    mse_tensor = torch.sum(torch.square(pred_matte - gt_matte)) / (
                            config["hyper_parameter"]["size"]["height"] * config["hyper_parameter"]["size"]["width"])
                    mse += mse_tensor.item()

                    mad_tensor = torch.sum(torch.abs(pred_matte - gt_matte)) / (
                            config["hyper_parameter"]["size"]["height"] * config["hyper_parameter"]["size"]["width"])
                    mad += mad_tensor.item()

                    n_sample += image.shape[0]
                avg_mse = mse / n_sample
                avg_mad = mad / n_sample
                if avg_mse < best_mse and avg_mad < best_mad:
                    torch.save(modnet.state_dict(), f"weights/best_weights_{config['training']['summary_writer']}.ckpt")
                    best_mse = avg_mse
                    best_mad = avg_mad
                print(f"epoch: {epoch} -- average mad: {avg_mse} -- average mda : {avg_mad}")
                writer.add_scalar("Validation-PPM100/MSE", avg_mse, epoch)
                writer.add_scalar("Validation-PPM100/MAD", avg_mad, epoch)

    # Predict image using best weights
    best_weights = torch.load(f"weights/best_weights_{config['training']['summary_writer']}.ckpt")
    modnet.load_state_dict(best_weights)
    with torch.no_grad():
        modnet.eval()
        for idx, file in enumerate(os.listdir("datasets/Evidence/")):
            file = os.path.join("datasets/Evidence", file)
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_crop = cv2.resize(image,
                                    (config["hyper_parameter"]["size"]["width"],
                                     config["hyper_parameter"]["size"]["height"]),
                                    interpolation=cv2.INTER_NEAREST)
            image_PIL = Image.fromarray(image_crop)
            img_tensor = transforms.ToTensor()(image_PIL)
            img_tensor = img_tensor.to("cuda:0")
            img_normalize_tensor = torch_transforms(image_PIL)
            img_normalize_tensor = img_normalize_tensor.unsqueeze(dim=0)
            _, _, pred_matte = modnet(img_normalize_tensor)
            pred_matte = pred_matte.squeeze(dim=0)
            pred_matte = pred_matte.repeat(3, 1, 1)
            result = torch.cat((img_tensor, pred_matte), dim=2)
            result_grid = make_grid(result)
            writer.add_image(f"{config['training']['summary_writer']}/image", result_grid, idx)
    writer.close()
