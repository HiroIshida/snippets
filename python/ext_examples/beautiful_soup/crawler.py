from dataclasses import dataclass, asdict
import certifi
from typing import List, Optional, Set, Tuple, NamedTuple, Dict
import re
import pathlib
from urllib import request
from bs4 import BeautifulSoup
from html.parser import HTMLParser
import subprocess

import ssl

def get_html_str(url: str):
    headers = { "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0" }
    print("requesting {} ...".format(url))
    req = request.Request(url, headers=headers)
    html_obj = request.urlopen(req, cafile=certifi.where())
    html_str = html_obj.read().decode("utf-8")
    return html_str


@dataclass
class DataInfo:
    url: str
    categories: List[str]


def updaqte_dataset_metadata(metadata: Dict[str, DataInfo], category: str, n_max_per_category: Optional[int] = 50):
    print("query: {}".format(category))

    base_url = "https://ambientcg.com/list?category=&date=&createdUsing=&basedOn=&q={}&method=&type=Material&sort=Popular".format(category)

    html_str = get_html_str(base_url)
    soup = BeautifulSoup(html_str, "html.parser") 

    hoge = soup.find("div", class_="ResultsContainer")
    url_dict = {}
    for elem in hoge.children:
        try:
            material_id = elem["id"]
            url = "https://ambientcg.com/view?id={}".format(material_id)
            url_dict[material_id] = url
        except:
            pass

    extension = "jpg"
    resolution = "2K"

    count = 0
    for material_id, sub_url in url_dict.items():

        if material_id in metadata:
            metadata[material_id].categories.append(category)
            continue

        sub_html_str = get_html_str(sub_url)
        soup = BeautifulSoup(sub_html_str, "html.parser")
        download_buttons = soup.find_all("a", class_="DownloadButton")

        for button in download_buttons:
            download_url = button["href"]

            if not extension.upper() in download_url:
                continue

            m = re.findall(r"(\w+)_(\w+)-{}.zip".format(extension.upper()), download_url)
            if m[0][1] != resolution:
                continue

            if extension.upper() in download_url and resolution.upper() in download_url:
                metadata[material_id] = DataInfo(download_url, [category])
        count += 1
        if n_max_per_category is not None:
            if count == n_max_per_category:
                return


#category_list = ["wooden", "fabric", "floor", "carpet", "stone", "metal", "dirty"]
category_list = ["wooden", "fabric", "floor", "carpet"]
metadata = {}

for category in category_list:
    updaqte_dataset_metadata(metadata, category)

metadata_alldict = {}
for key, val in metadata.items():
    metadata_alldict[key] = asdict(val)

datadir = pathlib.Path("./texture")
datadir.mkdir(exist_ok=True)

metadata_path = datadir / "metadata.yaml"

import yaml
with metadata_path.open(mode = "w") as f:
    yaml.dump(metadata_alldict, f)

#from IPython import embed; embed()


#output_file_name = "{}-{}.zip".format(material_id, resolution.lower())
#zip_full_file_path = datadir / output_file_name
#extract_full_file_path = datadir / material_id
#extract_full_file_path.mkdir(exist_ok=True)
#subprocess.run(["wget", "-c", "--no-check-certificate", download_url, "-O", str(zip_full_file_path)])
#subprocess.run(["unzip", str(zip_full_file_path), "-d", str(extract_full_file_path)])
#subprocess.run(["rm", str(zip_full_file_path)])
