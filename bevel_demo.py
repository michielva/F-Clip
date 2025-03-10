import glob
import json
from datetime import datetime
import numpy as np
import os
import time
import cv2 as cv
import torch
from bevel_test import build_model, C, M
import matplotlib.pyplot as plt


# variables
threshold = 0.4
model_name = 'HG2_LB'
extra_annotation = 'valid_set'
input_dir = '/nas/UnivisionAI/development/bevel/data/raw/valid'
output_dir = '/nas/UnivisionAI/development/bevel/fclp/output/inference'
ckpt = '/nas/UnivisionAI/development/bevel/fclp/output/training/230830-140929-HG2_LB/checkpoint_best.pth.tar'


class ImageStreamer:
    """Streams grayscale images from a directory"""
    def __init__(self, input_directory, img_glob):
        # list images in the input directory that satisfy img_glob
        print('\n==> Loading image paths.')
        search = os.path.join(input_directory, img_glob)
        self.list_img_paths = glob.glob(search)
        self.amount_images = len(self.list_img_paths)

        # no images are found
        if len(self.list_img_paths) == 0:
            raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')
        else:
            print(f'==> Successfully loaded {self.amount_images} image paths.')

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.amount_images:
            # todo: grayscale
            # image = cv.imread(self.list_img_paths[self.index], cv.IMREAD_GRAYSCALE)
            image = cv.imread(self.list_img_paths[self.index])
            image_name = os.path.basename(self.list_img_paths[self.index])
            self.index += 1
            return image, image_name
        else:
            raise StopIteration


class FClipDetect:
    def __init__(self, modeluse, ckpt=None):

        print(f'\n==> Loading {modeluse} model from checkpoint file {ckpt}')

        # load base config file & update it with model specific content
        if modeluse in ['HG1_D3', 'HG1', 'HG2', 'HG2_LB', 'HR']:
            config_file = f'config/fclip_{modeluse}.yaml'
        else:
            raise ValueError('Incorrect model was given.')
        print(f'==> Using config file: {config_file}')
        C.update(C.from_yaml(filename='config/base.yaml'))
        C.update(C.from_yaml(filename=config_file))
        M.update(C.model)
        C.io.model_initialize_file = ckpt

        # set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        iscpu = False if torch.cuda.is_available() else True
        print('==> Using device: ', self.device)

        # load model
        self.model = build_model(cpu=iscpu)
        self.model.to(self.device)
        print('==> Successfully loaded model.\n')

        # get input resolution, mean & standard deviation of images
        self.input_resolution = (M.resolution * 4, M.resolution * 4)
        self.image_mean = M.image.mean
        self.image_stddev = M.image.stddev

        # load dummy image
        # dummy_img =

        # export model to onnx



    def detect(self, img):
        # resize image
        orig_height, orig_width = img.shape[:2]
        input_img = cv.resize(img, self.input_resolution)
        input_img = input_img[:, :, ::-1]  # BGR to RGB
        resized_height, resized_width, channels = input_img.shape

        # normalize input
        input_img = (input_img.astype(np.float32) - self.image_mean) / self.image_stddev
        input_img = torch.from_numpy(input_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device=self.device)
        input_dict = {"image": input_img}

        # run inference for input image
        with torch.no_grad():
            outputs = self.model(input_dict, isTest=True)
            lines = outputs["heatmaps"]["lines"][0][:5] * 4
            scores = outputs["heatmaps"]["score"][0][:5]
            # lines = lines[score > threshold]

        # calculate back to original size
        lines[:, :, 0] = lines[:, :, 0] * orig_height / resized_height
        lines[:, :, 1] = lines[:, :, 1] * orig_width / resized_width
        if torch.cuda.is_available():
            lines = lines.cpu().numpy()
            scores = scores.cpu().numpy()
        else:
            lines = lines.numpy()
            scores = scores.numpy()

        return lines, scores


if __name__ == '__main__':

    # make output directories
    output_dir = output_dir + f'/{datetime.now().strftime("%d%m%Y_%H%M%S")}_{model_name}_{extra_annotation}'
    os.makedirs(output_dir, exist_ok=True)

    # save parameters in json file
    parameters = {
        'model': model_name,
        'input_dir': input_dir,
        'ckpt_used': ckpt,
        'threshold': threshold
    }
    with open(f'{output_dir}/parameters.json', 'w') as f:
        json.dump(parameters, f)

    # load image streamer
    image_stream = ImageStreamer(input_directory=input_dir, img_glob='*.png')

    # load model
    detector = FClipDetect(model_name, ckpt)

    # for img in image_stream:

    # run demo on images
    t_begin = time.time()
    print('\n==> Running demo.')
    for i, (img, img_name) in enumerate(image_stream):
        print(f'Image {i+1}: {img_name}')
        lines, scores = detector.detect(img)

        output_img = img
        for i, line in enumerate(lines):
            h, w, _ = output_img.shape
            start_coord = (int(lines[i][0][1]), int(lines[i][0][0]))
            end_coord = (int(lines[i][1][1]), int(lines[i][1][0]))
            if scores[i] > threshold:
                color = (5, 133, 41)
                position = (start_coord[0]+10, start_coord[1]-10)
            else:
                color = (9, 3, 173)
                position = (end_coord[0]-50, end_coord[1]-10)

            position = (max(min(position[0], w-100), 100), max(min(position[1], h-50), 50))

            cv.line(output_img, start_coord, end_coord, color, 1, lineType=16)
            cv.putText(output_img, text=f'{scores[i]:.2f}', org=position, color=color, thickness=2,
                       fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1, lineType=16)
            cv.imwrite(f"{output_dir}/{img_name}_test.png", output_img)

    t_end = time.time()
    print(f'Total time spent: {t_end - t_begin} s')
    print(f'Average frame rate: {image_stream.amount_images / (t_end - t_begin)} frames/s')

    print('==> Finished Demo.')
