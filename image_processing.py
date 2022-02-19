import cv2.cv2 as cv2
from MODNet.src.models.backbones.wrapper import MobileNetV2Backbone
import torch
from torchvision import transforms

torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

def load_mobilenetv2_human_seg():
    backbone = MobileNetV2Backbone(in_channels=3)
    backbone.load_pretrained_ckpt()
    return backbone


if __name__ == '__main__':
    img = cv2.imread("datasets/PPM-100/image/6146816_556eaff97f_o.jpg")
    # dst = cv2.ximgproc.guidedFilter(img, matte, 33, 2, -1)
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    mobile_net_v2 = load_mobilenetv2_human_seg()
    with torch.no_grad():
        mobile_net_v2.eval()
        img_tensor = transforms.ToTensor()(img)
        img_tensor = img_tensor.unsqueeze(dim=0)
        output = mobile_net_v2(img_tensor)
    # cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
    # cv2.imshow("img", img)
    # cv2.imshow("output", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(output)
