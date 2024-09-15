import cv2
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
import numpy as np


class AveragingQueue:
    def __init__(self, size):
        self.size = size
        self.queue = []
    def push(self, value):
        self.queue.append(value)
        if len(self.queue) > self.size:
            self.queue.pop(0)
    def get_average(self):
        return sum(self.queue) / len(self.queue)


if __name__ == "__main__":
  mp_drawing = mp.solutions.drawing_utils
  mp_hands = mp.solutions.hands
  cap = cv2.VideoCapture(0)

  thumbs_tip_queue = AveragingQueue(5)
  thumbs_knuckle_queue = AveragingQueue(5)
  index_fingers_tip_queue = AveragingQueue(5)
  index_fingers_knuckle_queue = AveragingQueue(5)

  toggle = False
  global_latched_point = np.array([0, 0])
  global_pos = None

  with mp_hands.Hands(
      static_image_mode=False,
      max_num_hands=2,
      min_detection_confidence=0.7,
      min_tracking_confidence=0.7
  ) as hands:
      while cap.isOpened():
          success, image = cap.read()
          if not success:
              print("Ignoring empty camera frame.")
              continue
          image = cv2.flip(image, 1)
          image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          results = hands.process(image_rgb)
          if results.multi_hand_landmarks:
              for hand_landmarks in results.multi_hand_landmarks:
                  mp_drawing.draw_landmarks(
                      image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

              coords_list = results.multi_hand_landmarks[0].landmark
              points = np.array([[landmark.x, landmark.y] for landmark in coords_list])
              # see this for indices
              # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
              thumbs_tip_queue.push(points[4])
              index_fingers_tip_queue.push(points[8])
              thumbs_knuckle_queue.push(points[3])
              index_fingers_knuckle_queue.push(points[7])

              reference_distance = np.linalg.norm(thumbs_knuckle_queue.get_average() - index_fingers_knuckle_queue.get_average())
              pointer = 0.5 * (thumbs_tip_queue.get_average() + index_fingers_tip_queue.get_average())

              pos_thumbs_tip = thumbs_tip_queue.get_average()
              pos_index_fingers_tip = index_fingers_tip_queue.get_average()
              dist_tip = np.linalg.norm(pos_thumbs_tip - pos_index_fingers_tip)
              if dist_tip < 0.05:
                  if not toggle:
                      toggle = True
                      print("start tracking")
                      latched_point = pointer
              if dist_tip > 0.05:
                  if toggle:
                      toggle = False
                      print("stop tracking")
                      latched_point = None
                      global_latched_point = global_pos
              if toggle:
                  relative_pos = pointer - latched_point
                  global_pos = global_latched_point + relative_pos
                  print(global_pos)
                  

              # dist_tip = ((thumbs_tip.x - index_fingers_tip.x)**2 + (thumbs_tip.y - index_fingers_tip.y)**2)**0.5
              # if dist_tip < 0.05:
              #   print("tracking")

          cv2.imshow('Hand Tracker', image)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break

  cap.release()
  cv2.destroyAllWindows()
