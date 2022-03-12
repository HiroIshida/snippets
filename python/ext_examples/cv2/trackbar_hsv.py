from dataclasses import dataclass
from copy import deepcopy
import cv2
import numpy as np
from typing import List

@dataclass
class SingleFilter:
    b_min: int
    b_max: int
    idx: int

    def get_logical(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.logical_and(hsv[:, :, self.idx] >= self.b_min, hsv[:, :, self.idx] <= self.b_max)


class HSVFilter:
    BLACK = (0, 0, 0)
    bound_list: List[SingleFilter]

    def __init__(self):
        h_bound = SingleFilter(0, 255, 0)
        s_bound = SingleFilter(0, 255, 1)
        v_bound = SingleFilter(0, 255, 2)
        self.bound_list = [h_bound, s_bound, v_bound]

    def get_logical(self, img: np.ndarray):
        logical = np.logical_and(*[bound.get_logical(img) for bound in self.bound_list])
        return logical

    def __call__(self, img: np.ndarray, fill_value=BLACK):
        fill_indices = np.logical_not(self.get_logical(img))
        img_out = deepcopy(img)
        img_out[fill_indices] = fill_value
        return img_out


if __name__ == '__main__':
    img = cv2.imread('data/cook.jpg')
    cv2.namedWindow('window')

    hsvfilter = HSVFilter()
    img_filtered = hsvfilter(img)

    import matplotlib.pyplot as plt 
    plt.imshow(img_filtered)
    plt.show()
