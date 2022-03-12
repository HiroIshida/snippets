from dataclasses import dataclass
from copy import deepcopy
import cv2
import numpy as np
from typing import List


@dataclass
class SingleFilter:
    name: str
    b_min: int
    b_max: int
    idx: int

    def get_logical(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.logical_and(hsv[:, :, self.idx] >= self.b_min, hsv[:, :, self.idx] <= self.b_max)

    def track_bar_min_name(self):
        return self.name + '_min'

    def track_bar_max_name(self):
        return self.name + '_max'

    def __post_init__(self):
        print(self.track_bar_min_name())
        cv2.createTrackbar(self.track_bar_min_name(), "window", 0, 255, lambda x: None)
        cv2.createTrackbar(self.track_bar_max_name(), "window", 0, 255, lambda x: None)

    def reflect_trackbar(self):
        self.b_min = cv2.getTrackbarPos(self.track_bar_min_name(), "window")
        self.b_max = cv2.getTrackbarPos(self.track_bar_max_name(), "window")


class HSVFilter:
    BLACK = (0, 0, 0)
    bound_list: List[SingleFilter]

    def __init__(self):
        cv2.namedWindow('window')
        h_bound = SingleFilter('h', 0, 255, 0)
        s_bound = SingleFilter('s', 0, 255, 1)
        v_bound = SingleFilter('v', 0, 255, 2)
        self.bound_list = [h_bound, s_bound, v_bound]

    def get_logical(self, img: np.ndarray):
        a = self.bound_list[0].get_logical(img)
        b = self.bound_list[1].get_logical(img)
        c = self.bound_list[2].get_logical(img)
        return a * b * c
    
    def reflect_trackbar(self):
        for bound in self.bound_list:
            bound.reflect_trackbar()

    def __call__(self, img: np.ndarray, fill_value=BLACK):
        fill_indices = np.logical_not(self.get_logical(img))
        img_out = deepcopy(img)
        img_out[fill_indices] = fill_value
        return img_out


if __name__ == '__main__':
    img = cv2.imread('data/cook.jpg')

    hsvfilter = HSVFilter()

    while True:
        cv2.imshow('window', hsvfilter(img))
        hsvfilter.reflect_trackbar()
        cv2.waitKey(10)
