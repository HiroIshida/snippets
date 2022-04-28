#!/usr/bin/env python3
import os
from dataclasses import dataclass
import xml.etree.ElementTree as ElementTree
from xml.etree.ElementTree import Element
from typing import Callable, Any, Tuple


def do_anything(tree, key: str, hook: Callable[[Element], Any]):
    for child in list(tree.getroot()):
        for gchild in child:
            if 'k' not in gchild.attrib:
                continue
            if gchild.attrib['k'] == key:
                hook(gchild)


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

    do_anything(tree, 'local_x', x_hook)
    do_anything(tree, 'local_y', y_hook)

    assert cdata.x_count == cdata.y_count

    x_mean = cdata.x_sum / cdata.x_count
    y_mean = cdata.y_sum / cdata.y_count
    return x_mean, y_mean


def rescale_map(tree):

    alpha = 3.0
    x_mean, y_mean = compute_mean_point(tree)

    def x_hook(e: Element):
        x_new = x_mean + alpha * (float(e.attrib['v']) - x_mean)
        e.attrib['v'] = str(x_new)

    def y_hook(e: Element):
        y_new = y_mean + alpha * (float(e.attrib['v']) - y_mean)
        e.attrib['v'] = str(y_new)

    def width_hook(e: Element):
        width_new = float(e.attrib['v']) * alpha
        e.attrib['v'] = str(width_new)

    do_anything(tree, 'local_x', x_hook)
    do_anything(tree, 'local_y', y_hook)
    do_anything(tree, 'width', width_hook)

tree = ElementTree.parse('ishida_play.osm')
ElementTree.dump(tree)
rescale_map(tree)
tree.write(os.path.expanduser('~/tmp/rescaled.osm'), encoding='UTF-8', xml_declaration=True)
