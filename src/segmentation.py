
from torchvision import transforms
from catalyst.utils import mask_to_overlay_image

import cv2
import torch
import numpy as np
import logging
logging.getLogger().setLevel(logging.DEBUG)


class RoadSegmentationNetwork:
    def __init__(self, model_path: str, input_shape=(1024, 576)):
        logging.info(f"loading model from {model_path}")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = torch.load(model_path)
            self.model = self.model.cuda()
        else:
            self.device = torch.device('cpu')
            self.model = torch.load(model_path, map_location=torch.device('cpu'))

        self.model.eval()
        self.input_shape = input_shape
        self.segm_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        logging.info("model loaded")

    def predict(self, drone_img_bgr: np.array, rotate: bool = False):
        image = cv2.cvtColor(drone_img_bgr, cv2.COLOR_BGR2RGB)
        if rotate:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.resize(image, self.input_shape)
        input_data = self.segm_transform(image)

        input_data = input_data.to(self.device)
        fdata = torch.unsqueeze(input_data,0).float()
        pred = self.model(fdata)
        cls5 = pred.cpu().detach().numpy()
        road_mask = cls5[0][1]
        road_mask = (road_mask * 255).astype(np.uint8)
        return road_mask
