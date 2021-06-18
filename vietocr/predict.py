import os
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="foo help")
    parser.add_argument("--config", required=True, help="foo help")

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    detector = Predictor(config)

    # Option for predicting folder images
    img_list = os.listdir(args.img)
    img_list = sorted(img_list)

    f_pre = open("./test_seq.txt", "w+")

    # new output <name>\t<gtruth>\t<predict>
    # f_gt = open("./gt_word.txt", "r")
    # lines = [line.strip("\n") for line in f_gt if line != "\n"]

    # start_time = time.time()
    # for img in lines:
    #     name, gt = img.split("\t")
    #     img_path = args.img + name
    #     image = Image.open(img_path)

    #     s, prob = detector.predict(image, return_prob=True)

    #     res = name + "\t" + gt + "\t" + s + "\t" + str(prob) + "\n"
    #     f_pre.write(res)
    # runtime = time.time() - start_time
    # print("FPS:", len(img_list) / runtime)

    start_time = time.time()
    for img in img_list:
        img_path = args.img + img
        image = Image.open(img_path)

        s = detector.predict(image)
        # print(img_path, "-----", s)

        res = img + "\t" + s + "\n"
        f_pre.write(res)
    runtime = time.time() - start_time
    print("FPS:", len(img_list) / runtime)


if __name__ == "__main__":
    main()
