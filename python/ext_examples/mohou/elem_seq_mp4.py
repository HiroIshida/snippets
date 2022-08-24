import os
import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip
from mohou.types import EpisodeBundle
from mohou.file import get_project_path
from mohou.types import RGBImage
import pickle
import cv2


def bench(seq, seq_again):
    print("len mismatch {}".format(len(seq) - len(seq_again)))
    diff_sum = 0
    for a, b in zip(seq, seq_again):
        size = np.prod(a.shape[:2])
        diff_per_pixel = np.linalg.norm(a - b) / size
        diff_sum += diff_per_pixel
    diff_mean = diff_sum / len(seq)
    print("error {}".format(diff_mean))


pp = get_project_path("bunsetsu_test_project")
bundle = EpisodeBundle.load(pp)
episode = bundle[0]
rgb_seq = episode.get_sequence_by_type(RGBImage)
seq = [e.numpy() for e in rgb_seq.elem_list]
size = tuple(rgb_seq.elem_shape[:2])

fps = 30

def test_moviepy_mp4():
    name = "hoge.mp4"
    clip = ImageSequenceClip(seq, fps=fps)
    clip.write_videofile(name, audio=False)
    clip_again = VideoFileClip(name)
    seq_again = [f for f in clip_again.iter_frames()]
    print("mp4 moviepy")
    print("size: {}".format(os.path.getsize(name)))
    bench(seq, seq_again)


def test_moviepy_gif():
    name = "hoge.gif"
    clip = ImageSequenceClip(seq, fps=fps)
    clip.write_gif(name)
    clip_again = VideoFileClip(name)
    seq_again = [f for f in clip_again.iter_frames()]
    print("gif moviepy")
    print("size: {}".format(os.path.getsize(name)))
    bench(seq, seq_again)

def test_opencv_mp4():
    name = "hoge-cv.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name, fourcc, fps, size)
    for arr in seq:
        out.write(arr)
    out.release()
    print("mp4 cv2...")
    print("size: {}".format(os.path.getsize(name)))

    cap = cv2.VideoCapture(name)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    bench(seq, frames)


def test_npy():
    name = "hoge.npy"
    print("testing npy..")
    np.save(name, np.array(seq))
    print("size: {}".format(os.path.getsize(name)))


test_opencv_mp4()
test_moviepy_mp4()
test_moviepy_gif()
test_npy()
