import numpy as np
from mohou.types import ElementDict, AngleVector, RGBImage
from mohou.file import get_project_path
from mohou.propagator import LSTMPropagator, LSTMPropagator
from tunable_filter.composite_zoo import HSVBlurCropResolFilter

def subscribe_image():
    return np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)


def subscribe_important_6_vector():
    return np.random.randn(6)


def send_to_robot(av: AngleVector):
    print(av.numpy())


if __name__ == "__main__":
    pp = get_project_path("iwata")
    model = LSTMPropagator.create_default(pp)
    model.set_device("cuda")
    image_filter = HSVBlurCropResolFilter.from_yaml(pp / "image_config.yaml")

    # feedback loop
    for _ in range(1000):
        image = subscribe_image()
        vector = subscribe_important_6_vector()

        av = AngleVector(vector)
        image = RGBImage(image_filter(image))
        elemd = ElementDict([av, image])
        model.feed(elemd)
        pred_elemd_list = model.predict(1)
        pred_elemd = pred_elemd_list[0]
        next_av = pred_elemd[AngleVector]
        send_to_robot(next_av)

        next_image = pred_elemd[RGBImage]  # predicted by vae + lstm. you can use it for debug
