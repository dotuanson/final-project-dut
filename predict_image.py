import cv2.cv2 as cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from MODNet.src.models.modnet import MODNet

torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

print('Load pre-trained MODNet...')
pretrained_ckpt = 'MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

GPU = True if torch.cuda.device_count() > 0 else False
if GPU:
    print('Use GPU...')
    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
else:
    print('Use CPU...')
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

modnet.eval()

if __name__ == '__main__':
    frame_np = cv2.imread("datasets/PPM-100/image/6146816_556eaff97f_o.jpg")
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
    frame_np = frame_np[:, 120:792, :]

    frame_np = cv2.flip(frame_np, 1)

    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = torch_transforms(frame_PIL)
    frame_tensor = frame_tensor[None, :, :, :]
    if GPU:
        frame_tensor = frame_tensor.cuda()

    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, True)

    # matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    # fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
    # view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
    # view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)
    print(matte_np.shape)
    cv2.imshow('output modnet', matte_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
