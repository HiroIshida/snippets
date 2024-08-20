import matplotlib.pyplot as plt
import numpy as np
import os

def create_image(index, height=100, width=100):
    np.random.seed(index)
    base_pattern = np.outer(
        np.sin(np.linspace(0, np.pi, height)),
        np.cos(np.linspace(0, np.pi, width))
    )
    noise = np.random.normal(0, 0.1, base_pattern.shape)
    image = base_pattern + noise
    image = np.clip(image, 0, 1)
    return image

def show_image_until_click(image):
    coordinates = []
    while not coordinates:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')

        def onclick(event):
            if event.button == 1:
                coordinates.append((event.xdata, event.ydata))
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    return coordinates

def get_user_decision():
    while True:
        decision = input("Confirm the selection? (y/n) or quit (q): ").strip().lower()
        if decision in ['y', 'n', 'q']:
            return decision

def save_annotated_image(image, coordinates, index, output_dir):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    if coordinates:
        x, y = coordinates[-1]
        ax.plot(x, y, 'ro')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/annotated_image_{index}.png")
    plt.close(fig)

def process_images(num_images):
    for index in range(num_images):
        while True:
            image = create_image(index)
            coordinates = show_image_until_click(image)
            click_coordinates.append(coordinates)
            save_annotated_image(image, coordinates, index, "./annotated")
            user_decision = get_user_decision()

            if user_decision == 'y':
                break
            elif user_decision == 'q':
                return
            else:
                click_coordinates.pop()

click_coordinates = []
process_images(3)
print("All Collected Coordinates:", click_coordinates)
