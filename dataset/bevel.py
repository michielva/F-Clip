"""
Process Bevel dataset (labeled with supervisely)
Usage:
    dataset/bevel.py <src> <dst>
    dataset/bevel.py (-h | --help )

Examples:
    python dataset/bevel.py /nas/UnivisionAI/development/bevel/data/raw /nas/UnivisionAI/development/bevel/data/raw/lines

Arguments:
    <src>                original directory
    <dst>                directory of the output

Options:
   -h --help             Show this screen.
"""

import os
import sys
import glob
import json
from pathlib import Path
from itertools import combinations

import cv2
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.io import loadmat
from scipy.ndimage import zoom

try:
    sys.path.append(".")
    sys.path.append("..")
    from FClip.utils import parmap
except Exception:
    raise


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def extract_line_information(ann) -> np.array:
    """
    Extract line information from supervisely format labels & convert to array of shape (n, 2, 2)

    :param ann: annotation json object
    :return: numpy array of shape (n, 2, 2) with n being the amount of annotated lines
    """

    lines = []

    for obj in ann['objects']:
        if obj['geometryType'] == 'line':
            lines.append(obj['points']['exterior'])

    return np.array(lines, dtype=np.float32)


def save_heatmap(path, image, lines, data_dst):
    # image name
    prefix = os.path.split(path)[1]

    # input image size & heatmap size
    img_rescale = (512, 512)
    heatmap_scale = (128, 128)

    # calculate resize factors & set up heatmaps (junction map, junction offset, line map)
    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    jmap = np.zeros((1,) + heatmap_scale, dtype=np.float32)
    joff = np.zeros((1, 2) + heatmap_scale, dtype=np.float32)
    lmap = np.zeros(heatmap_scale, dtype=np.float32)

    # clip line values & flip x and y coordinates
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    lines = lines[:, :, ::-1]

    # junction & line map
    junc = []
    jids = {}
    def jid(jun):
        jun = tuple(jun[:2])
        if jun in jids:
            return jids[jun]
        jids[jun] = len(junc)
        junc.append(np.array(jun + (0,)))
        return len(junc) - 1

    lnid = []
    lpos, lneg = [], []
    for v0, v1 in lines:
        lnid.append((jid(v0), jid(v1)))
        lpos.append([junc[jid(v0)], junc[jid(v1)]])

        vint0, vint1 = to_int(v0), to_int(v1)
        jmap[0][vint0] = 1
        jmap[0][vint1] = 1
        rr, cc, value = skimage.draw.line_aa(*to_int(v0), *to_int(v1))
        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

    for v in junc:
        vint = to_int(v[:2])
        joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5

    llmap = zoom(lmap, [0.5, 0.5])
    lineset = set([frozenset(l) for l in lnid])
    for i0, i1 in combinations(range(len(junc)), 2):
        if frozenset([i0, i1]) not in lineset:
            v0, v1 = junc[i0], junc[i1]
            vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)
            rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
            lneg.append([v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))])

    # assert len(lneg) != 0
    # lneg.sort(key=lambda l: -l[-1])

    # junctions & line information
    junc = np.array(junc, dtype=np.float32)
    Lpos = np.array(lnid, dtype=np.int64)
    Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=np.int64)
    lpos = np.array(lpos, dtype=np.float32)
    lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)

    # resize image
    image = cv2.resize(image, img_rescale)

    # save images
    Path(data_dst).joinpath('plots').mkdir(parents=False, exist_ok=True)
    plt.imshow(image)
    for i0, i1 in Lpos:
        plt.scatter(junc[i0][1] * 4, junc[i0][0] * 4)
        plt.scatter(junc[i1][1] * 4, junc[i1][0] * 4)
        plt.plot([junc[i0][1] * 4, junc[i1][1] * 4], [junc[i0][0] * 4, junc[i1][0] * 4])
    plt.savefig(Path(data_dst).joinpath(f'plots/{prefix}_gt_line.png'))
    plt.close()

    # save _label.npz file
    np.savez_compressed(
        Path(data_dst).joinpath(f'{prefix}_label.npz'),
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]
        joff=joff,  # [J, 2, H, W]
        lmap=lmap,  # [H, W]
        junc=junc,  # [Na, 3]
        Lpos=Lpos,  # [M, 2]
        Lneg=Lneg,  # [M, 2]
        lpos=lpos,  # [Np, 2, 3]   (y, x, t) for the last dim
        lneg=lneg,  # [Nn, 2, 3]
    )
    cv2.imwrite(str(Path(data_dst).joinpath(f'{prefix}.png')), image)


def main():
    try:
        # get args from command line arguments
        args = docopt(__doc__)
        data_root = args["<src>"]
        data_output = args["<dst>"]
    except:
        # default values when file is ran without cli arguments
        data_root = '/nas/UnivisionAI/development/bevel/data/raw'
        data_output = '/nas/UnivisionAI/development/bevel/data/raw/lines'
    os.makedirs(data_output, exist_ok=True)

    dataset = sorted(glob.glob(os.path.join(data_root, "img/*.png")))

    def handle(img_path):
        # get image
        prefix = os.path.split(img_path)[1].replace(".png", "")
        img = cv2.imread(img_path)

        # get annotation
        ann_path = os.path.join(data_root, f'ann/{prefix + ".png.json"}')
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        # extract all lines for image in array format (n, 2, 2)
        lines = extract_line_information(ann)

        # convert to _label file (containing junction map, line map,
        path = os.path.join(data_output, prefix)
        save_heatmap(f"{path}", img[::, ::], lines, data_output)
        print(f"Finishing {path}")

    parmap(handle, dataset)


if __name__ == "__main__":
    main()
