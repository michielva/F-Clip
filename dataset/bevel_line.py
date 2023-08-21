import numpy as np
import os
import cv2
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(path, image, lines, data_dst):
    # image name
    prefix = os.path.split(path)[1][:-10]

    # input image size & heatmap size
    img_rescale = (512, 512)
    heatmap_scale = (128, 128)

    # construct the empty heatmaps
    lcmap = np.zeros(heatmap_scale, dtype=np.float32)  # line center: (128, 128)
    lcoff = np.zeros((2,) + heatmap_scale, dtype=np.float32)  # line center offsets: (2, 128, 128)
    lleng = np.zeros(heatmap_scale, dtype=np.float32)  # line length: (128, 128)
    angle = np.zeros(heatmap_scale, dtype=np.float32)  # line angle: (128, 128)

    # calculate heatmaps for each ground truth line
    for v0, v1 in lines:
        # line center, line center offsets & line length
        v = (v0 + v1) / 2
        vint = to_int(v)
        lcmap[vint] = 1
        lcoff[:, vint[0], vint[1]] = v - vint - 0.5
        lleng[vint] = np.sqrt(np.sum((v0 - v1) ** 2)) / 2  # L

        # get starting point
        if v0[0] <= v[0]:
            vv = v0
        else:
            vv = v1

        # the angle under the image coordinate system (r, c)
        # theta means the component along the c direction on the unit vector
        if np.sqrt(np.sum((vv - v) ** 2)) <= 1e-4:
            continue
        angle[vint] = np.sum((vv - v) * np.array([0., 1.])) / np.sqrt(np.sum((vv - v) ** 2))  # theta

    # make folder & set default plot font size & colors
    Path(data_dst).joinpath('plots').mkdir(exist_ok=True, parents=False)
    plt.rcParams.update({'font.size': 30})
    colors_list = ['#c7c5c5', '#34eb6e']
    cmap = colors.ListedColormap(colors_list)

    # plot the ground truth heatmap for line center
    plt.figure(figsize=(30, 30))
    plt.imshow(lcmap, cmap=cmap, vmin=0, vmax=1)
    plt.title('Ground truth heatmap for line center (lcmap)', fontsize=36)
    plt.xlabel('X', fontsize=36)
    plt.ylabel('Y', fontsize=36)
    for y, x in np.transpose(np.where(lcmap > 0)):
        plt.annotate(str(lcmap[y, x]), xy=(x, y), ha='left', va='bottom', fontsize=24)
    plt.savefig(Path(data_dst).joinpath('plots', f'{prefix}_heatmap_lcmap.png'))
    plt.close()

    # plot the ground truth heatmap for line length
    plt.figure(figsize=(30, 30))
    plt.imshow(lleng, cmap=cmap, vmin=0, vmax=1)
    plt.title('Ground truth heatmap for line length (lleng) - actually 1/2 of total length', fontsize=36)
    plt.xlabel('X', fontsize=36)
    plt.ylabel('Y', fontsize=36)
    for y, x in np.transpose(np.where(lleng > 0)):
        plt.annotate(str(lleng[y, x]), xy=(x, y), ha='left', va='bottom', fontsize=24)
    plt.savefig(Path(data_dst).joinpath('plots', f'{prefix}_heatmap_lleng.png'))
    plt.close()

    # plot the ground truth heatmap for line angle
    plt.figure(figsize=(30, 30))
    # plt.imshow(angle, cmap=cmap, vmin=-1, vmax=1)
    plt.imshow(angle)
    plt.title('Ground truth heatmap for line angle', fontsize=36)
    plt.xlabel('X', fontsize=36)
    plt.ylabel('Y', fontsize=36)
    for y, x in np.transpose(np.where(angle != 0)):
        plt.scatter(x, y=y, color='green', s=40)
        plt.annotate(str(angle[y, x]), xy=(x, y), ha='left', va='bottom', fontsize=24)
    plt.savefig(Path(data_dst).joinpath('plots', f'{prefix}_heatmap_angle.png'))
    plt.close()

    # plot the ground truth heatmap for the offsets length
    plt.figure(figsize=(60, 30))
    plt.subplots(121)
    plt.imshow(lcoff[0], cmap=cmap, vmin=0, vmax=1)
    plt.title('Ground truth heatmap for line length (lleng) - actually 1/2 of total length', fontsize=36)
    plt.xlabel('X', fontsize=36)
    plt.ylabel('Y', fontsize=36)
    for y, x in np.transpose(np.where(lleng > 0)):
        plt.annotate(str(lleng[y, x]), xy=(x, y), ha='left', va='bottom', fontsize=24)
    plt.subplots(122)
    plt.savefig(Path(data_dst).joinpath('plots', f'{prefix}_heatmap_lleng.png'))
    plt.close()

    # coor = np.argwhere(lcmap == 1)
    # for yx in coor:
    #     offset = lcoff[:, int(yx[0]), int(yx[1])]
    #     length = lleng[int(yx[0]), int(yx[1])]
    #     theta = angle[int(yx[0]), int(yx[1])]
    #
    #     center = yx + offset
    #     d = np.array([-np.sqrt(1-theta**2), theta])
    #     plt.scatter(center[1]*4, center[0]*4, c="b")
    #
    #     plt.arrow(center[1]*4, center[0]*4, d[1]*length*4, d[0]*length*4,
    #               length_includes_head=True,
    #               head_width=15, head_length=25, fc='r', ec='b')

    # plt.savefig(f"{prefix}_line.png", dpi=200), plt.close()

    # plt.subplot(122), \
    # plt.imshow(image)
    # coor = np.argwhere(lcmap == 1)
    # for yx in coor:
    #     offset = lcoff[:, int(yx[0]), int(yx[1])]
    #     length = lleng[int(yx[0]), int(yx[1])]
    #     theta = angle[int(yx[0]), int(yx[1])]
    #
    #     center = yx + offset
    #     d = np.array([-np.sqrt(1-theta**2), theta])
    #
    #     n0 = center + d * length
    #     n1 = center - d * length
    #     plt.plot([n0[1] * 4, n1[1] * 4], [n0[0] * 4, n1[0] * 4])
    # plt.savefig(f"{prefix[-8:]}_line.png", dpi=100), plt.close()

    np.savez_compressed(
        f"{prefix}_line.npz",
        lcmap=lcmap,
        lcoff=lcoff,
        lleng=lleng,
        angle=angle,
    )


if __name__ == '__main__':

    data_root = '/nas/UnivisionAI/development/bevel/data/raw/lines'
    data_output = '/nas/UnivisionAI/development/bevel/data/raw/lines'

    filelist = glob.glob(f"{data_root}/*_label.npz")

    for file in filelist:
        with np.load(file) as npz:
            lines = npz["lpos"][:, :, :2]
        image = cv2.imread(file.replace("_label.npz", ".png"))
        save_heatmap(file, image, lines, data_output)
        print(f'Finishing {file}.')

