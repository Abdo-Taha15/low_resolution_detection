import cv2
import argparse
from imutils import paths
import os
import imutils

import torch
import torch.backends.cudnn as cudnn

import imgproc
from craft import CRAFT

from blur_detection import (
    detect_blur_fft,
    text_detection_score,
    text_recognition_score,
    copyStateDict,
)


def is_recognizable(
    net,
    image_path,
    blur_thresh,
    text_threshold,
    link_threshold,
    low_text,
    cuda,
    canvas_size,
    mag_ratio,
    rec,
):
    # load the input image from disk, resize it, and convert it to
    # grayscale
    orig = cv2.imread(image_path)
    # orig = imutils.resize(orig, width=500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # apply our blur detector using the FFT
    fft, is_blurry = detect_blur_fft(gray, size=60, thresh=blur_thresh)
    print(f"fft: {fft:.3f}, is_blurry: {is_blurry}")
    if is_blurry:
        return False

    if rec:
        _, is_rec, _ = text_recognition_score(orig, text_threshold)
        if is_rec:
            return True
    else:
        image = imgproc.loadImage(orig)
        size, score, size_len, is_rec = text_detection_score(
            net,
            image,
            text_threshold,
            link_threshold,
            low_text,
            cuda,
            canvas_size,
            mag_ratio,
        )
        print(f"size: {size:.3f}, score: {score:.3f}")
        if size_len and is_rec:
            return True
    return False


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", required=True, help="input folder path or image path"
    )
    parser.add_argument(
        "--blur_threshold", type=float, default=20, help="blur threshold"
    )
    parser.add_argument(
        "--trained_model",
        default="weights/craft_mlt_25k.pth",
        type=str,
        help="pretrained model",
    )
    parser.add_argument(
        "--text_threshold", default=0.8, type=float, help="text confidence threshold"
    )
    parser.add_argument(
        "--low_text", default=0.4, type=float, help="text low-bound score"
    )
    parser.add_argument(
        "--link_threshold", default=0.4, type=float, help="link confidence threshold"
    )
    parser.add_argument(
        "--cuda", default=True, type=bool, help="Use cuda for inference"
    )
    parser.add_argument(
        "--canvas_size", default=1280, type=int, help="image size for inference"
    )
    parser.add_argument(
        "--mag_ratio", default=1.5, type=float, help="image magnification ratio"
    )
    parser.add_argument("--rec", default=False, type=bool, help="Use OCR recognition")
    args = parser.parse_args()

    # load net
    net = CRAFT()  # initialize

    print(f"Loading weights from checkpoint ('{args.trained_model}')")
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(
            copyStateDict(torch.load(args.trained_model, map_location="cpu"))
        )

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    if os.path.isdir(args.image):
        # Get all image paths in the input folder
        image_paths = list(paths.list_images(args.image))

        # Process each image
        for image_path in image_paths:
            is_rec = is_recognizable(
                net,
                image_path,
                args.blur_threshold,
                args.text_threshold,
                args.link_threshold,
                args.low_text,
                args.cuda,
                args.canvas_size,
                args.mag_ratio,
                args.rec,
            )
            print(f"is {image_path} recognizable: {is_rec}")
    elif os.path.isfile(args.image):
        is_rec = is_recognizable(
            net,
            args.image,
            args.blur_threshold,
            args.text_threshold,
            args.link_threshold,
            args.low_text,
            args.cuda,
            args.canvas_size,
            args.mag_ratio,
            args.rec,
        )
        print(f"is {image_path} recognizable: {is_rec}")


if __name__ == "__main__":
    main()
