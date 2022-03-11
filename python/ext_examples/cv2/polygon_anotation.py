from copy import deepcopy

import cv2
import numpy as np
  
# path
path = './data/cook.jpg'

class PolygonDrawer:
    GREEN = (0, 200, 0)
    RED = (0, 0, 200)
    WHITE = (255, 255, 255)

    def __init__(self, image):
        self.image = image
        self.gui_image = deepcopy(image)
        self.mask_image = None
        self.polygons = []
        self.points = []
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        th_close_enouch = 10

        if len(self.points) == 0:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                cv2.circle(self.gui_image, (x, y), 5, self.GREEN, -1)
            return

        dist = np.linalg.norm(np.array(self.points[0]) - np.array([x, y]))

        if event == cv2.EVENT_MOUSEMOVE:
            if len(self.points) > 1:
                if dist < th_close_enouch:
                    cv2.circle(self.gui_image, self.points[0], 5, self.RED, -1)
                else:
                    cv2.circle(self.gui_image, self.points[0], 5, self.GREEN, -1)

        if event == cv2.EVENT_LBUTTONDOWN:

            if dist < th_close_enouch:
                x, y = self.points[0]
                cv2.fillPoly(self.gui_image, [np.array(self.points, 'int32')], color=self.GREEN)
            else:
                cv2.circle(self.gui_image, (x, y), 5, self.GREEN, -1)
                self.points.append((x, y))
                cv2.line(self.gui_image, self.points[-2], self.points[-1], color=self.GREEN)

    def reset(self):
        if len(self.points) > 0:
            self.polygons.append(np.array(self.points, 'int32'))
            self.points = []

    def create_mask(self):
        finish_flag = False
        while True:
            print('reset')
            self.reset()
            if finish_flag:
                print("going to finish")
                break
            while True:
                cv2.imshow("image", self.gui_image)
                if cv2.waitKey(50) == ord('q'):
                    finish_flag = True
                    break
                if cv2.waitKey(50) == ord('n'):
                    break

        mask_image = np.zeros(self.image.shape)
        cv2.fillPoly(mask_image, self.polygons, color=self.WHITE)
        return mask_image


image = cv2.imread(path)
pd = PolygonDrawer(image)
mask = pd.create_mask()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
indices = np.where(np.all(mask == (255, 255, 255), axis=-1))
hsv_mask = hsv[indices]
hsv_flatten = hsv.reshape(-1, 3)

import matplotlib.pyplot as plt
plt.scatter(hsv_flatten[:, 0], hsv_flatten[:, 1], c='blue')
plt.scatter(hsv_mask[:, 0], hsv_mask[:, 1], c='red', alpha=0.4)
plt.show()
