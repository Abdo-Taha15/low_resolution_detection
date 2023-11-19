import cv2
import argparse
from imutils import paths
import os
import imutils
from blur_detection import detect_blur_fft, text_score


def is_recognizable(
    image_path, blur_thresh: float = 20, text_score_thresh: float = 0.8
):
    # load the input image from disk, resize it, and convert it to
    # grayscale
    orig = cv2.imread(image_path)
    orig = imutils.resize(orig, width=500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # apply our blur detector using the FFT
    _, blurry = detect_blur_fft(gray, size=60, thresh=blur_thresh)
    if blurry:
        return False
    _, is_rec, _ = text_score(orig, text_score_thresh)
    if not is_rec:
        return False
    return True


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, help="input folder path or image path")
    parser.add_argument(
        "--blur_threshold", type=float, default=20, help="blur threshold"
    )
    parser.add_argument(
        "--text_score_threshold", type=float, default=0.8, help="text score threshold"
    )
    args = parser.parse_args()

    if os.path.isdir(args.i):
        # Get all image paths in the input folder
        image_paths = list(paths.list_images(args.i))

        # Process each image
        for image_path in image_paths:
            is_rec = is_recognizable(
                image_path, args.blur_threshold, args.text_score_threshold
            )
            if is_rec:
                print(f"is {image_path} recognizable: {is_rec}")
            else:
                print(f"is {image_path} recognizable: {is_rec}")
    elif os.path.isfile(args.i):
        is_rec = is_recognizable(args.i, args.blur_threshold, args.text_score_threshold)
        if is_rec:
            print(f"is {args.i} recognizable: {is_rec}")
        else:
            print(f"is {args.i} recognizable: {is_rec}")


if __name__ == "__main__":
    main()
