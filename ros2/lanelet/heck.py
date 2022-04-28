#!/usr/bin/env python3
import os
from dataclasses import dataclass
import xml.etree.ElementTree as ElementTree
from xml.etree.ElementTree import Element
from typing import Callable, Any, Tuple


def do_anything(tree, x_hook: Callable[[Element], Any], y_hook: Callable[[Element], Any]):
    for child in list(tree.getroot()):
        if child.tag == 'node':
            for gchild in child:
                assert list(gchild.keys()) == ['k', 'v']
                if gchild.attrib['k'] == 'local_x':
                    x_hook(gchild)
                if gchild.attrib['k'] == 'local_y':
                    y_hook(gchild)


def compute_mean_point(tree) -> Tuple[float, float]: 

    @dataclass
    class CaptureData:
        x_sum: float = 0.0
        y_sum: float = 0.0
        x_count: int = 0
        y_count: int = 0

    cdata = CaptureData()

    def x_hook(e: Element):
        cdata.x_sum += float(e.attrib['v'])
        cdata.x_count += 1

    def y_hook(e: Element):
        cdata.y_sum += float(e.attrib['v'])
        cdata.y_count += 1

    do_anything(tree, x_hook, y_hook)
    assert cdata.x_count == cdata.y_count

    x_mean = cdata.x_sum / cdata.x_count
    y_mean = cdata.y_sum / cdata.y_count
    return x_mean, y_mean


def rescale_map(tree):

    @dataclass
    class CaptureData:
        alpha: float
        x_mean: float
        y_mean: float

    alpha = 3.0
    cdata = CaptureData(alpha, *compute_mean_point(tree))

    def x_hook(e: Element):
        x_new = cdata.x_mean + cdata.alpha * (float(e.attrib['v']) - cdata.x_mean)
        e.attrib['v'] = str(x_new)

    def y_hook(e: Element):
        y_new = cdata.y_mean + cdata.alpha * (float(e.attrib['v']) - cdata.y_mean)
        e.attrib['v'] = str(y_new)
    do_anything(tree, x_hook, y_hook)

tree = ElementTree.parse('ishida_play.osm')
ElementTree.dump(tree)
rescale_map(tree)
tree.write(os.path.expanduser('~/tmp/rescaled.osm'), encoding='UTF-8', xml_declaration=True)
