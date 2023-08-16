import numpy as np
from pathlib import Path
from PIL import Image

path_label_york = '/nas/UnivisionAI/development/bevel/fclp/york/valid/P1020171_label.npz'
path_line_york = '/nas/UnivisionAI/development/bevel/fclp/york/valid/P1020171_line.npz'
path_image_york = '/nas/UnivisionAI/development/bevel/fclp/york/valid/P1020171.png'

label = np.load(path_label_york)
