import logging
import os
import torch
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image


def size_by_long_side(orig_size, long_side, align=32):
    # Scale so the longer side equals `long_side`, keep the aspect ratio, and
    # align both sides to `align` (32) as required by the model's downsampling.
    # `orig_size` and the return value are both (width, height).
    w, h = orig_size
    if w >= h:
        new_w = long_side
        new_h = max(align, round(h * long_side / w))
    else:
        new_h = long_side
        new_w = max(align, round(w * long_side / h))
    new_w = max(align, int(round(new_w / align)) * align)
    new_h = max(align, int(round(new_h / align)) * align)
    return (new_w, new_h)


def path_to_image(path, size=(1024, 1024), color_type=['rgb', 'gray'][0]):
    if color_type.lower() == 'rgb':
        image = cv2.imread(path)
    elif color_type.lower() == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print('Select the color_type to return, either to RGB or gray image.')
        return
    if size:
        # An int `size` means "scale the longer side to this value while keeping
        # the aspect ratio"; a (w, h) tuple keeps the original fixed-size behavior.
        if isinstance(size, int):
            h_orig, w_orig = image.shape[:2]
            size = size_by_long_side((w_orig, h_orig), size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if color_type.lower() == 'rgb':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    else:
        image = Image.fromarray(image).convert('L')
    return image



def check_state_dict(state_dict, unwanted_prefixes=['module.', '_orig_mod.']):
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        state_dict[k[prefix_length:]] = state_dict.pop(k)
    return state_dict


def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1-epsilon)*gts+epsilon/2
    return new_gts


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger('BiRefNet')
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
