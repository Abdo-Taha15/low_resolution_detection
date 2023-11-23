# import the necessary packages
import numpy as np
from paddleocr import PaddleOCR
import time

import torch
from torch.autograd import Variable

import cv2
import imgproc

from collections import OrderedDict


def detect_blur_fft(image, size=60, thresh: float = 20):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size : cY + size, cX - size : cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean < thresh)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def get_score(textmap, linkmap, link_threshold, low_text):
    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, _ = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )
    mean_size = np.mean([stats[k, cv2.CC_STAT_AREA] for k in range(1, nLabels)])
    mean_score = np.mean([np.max(textmap[labels == k]) for k in range(1, nLabels)])
    return mean_size, mean_score


def text_detection_score(
    net,
    image,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    cuda: bool = False,
    canvas_size: int = 1280,
    mag_ratio: float = 1.5,
):
    # resize
    img_resized, _, _ = imgproc.resize_aspect_ratio(
        image,
        canvas_size,
        interpolation=cv2.INTER_LINEAR,
        mag_ratio=mag_ratio,
    )

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, _ = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    mean_size, mean_score = get_score(score_text, score_link, link_threshold, low_text)

    return [mean_size, mean_score, mean_size >= 10, mean_score >= text_threshold]


def text_recognition_score(img, thresh: float = 0.8):
    start = time.time()

    ocr = PaddleOCR(use_angle_cls=True, lang="ch")

    result = ocr.ocr(img, cls=False)
    scores = [line[1][1] for line in result[0]]
    mean_score = np.mean(scores)

    end = time.time()
    interval = end - start
    return [mean_score, mean_score >= thresh, interval]
