import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

def show_image_for_box_annotation(image, n_boxes):
    box_coordinates = []
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    close_flag = [False]
    current_box = [None, None]
    
    def onclick(event):
        if event.button == 1:
            if current_box[0] is None:
                current_box[0] = (event.xdata, event.ydata)
                ax.plot(event.xdata, event.ydata, 'ro')
                fig.canvas.draw()
            else:
                current_box[1] = (event.xdata, event.ydata)
                x1, y1 = current_box[0]
                x2, y2 = current_box[1]
                width = x2 - x1
                height = y2 - y1
                rect = Rectangle((x1, y1), width, height, fill=False, edgecolor='red')
                ax.plot(x1, y1, 'ro')
                ax.add_patch(rect)
                box_coordinates.append((x1, y1, x2, y2))
                current_box[0] = None
                current_box[1] = None
                fig.canvas.draw()
                
                if len(box_coordinates) == n_boxes:
                    close_flag[0] = True

    def timer_callback():
        if close_flag[0]:
            time.sleep(0.5)
            plt.close(fig)

    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(timer_callback)
    timer.start()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    return box_coordinates

image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
boxes = show_image_for_box_annotation(image, 3)
print(boxes)
