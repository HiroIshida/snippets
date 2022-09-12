import pygame
import numpy as np
# https://github.com/ChristianD37/YoutubeTutorials/blob/master/PS4%20Controller/test.py
pygame.init()
pygame.joystick.init()
count = pygame.joystick.get_count()
assert count == 1
controller = pygame.joystick.Joystick(0)
controller.init()

while True:
    for e in pygame.event.get():
        if e.type == pygame.JOYAXISMOTION:
            vector = np.array([controller.get_axis(0), controller.get_axis(1)])
            print(vector)

    #if e.type == pygame.JOYBUTTONDOWN:
    val = controller.get_button(4)
    if val == 1:
        break
