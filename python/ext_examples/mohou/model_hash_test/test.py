import subprocess
import pickle
import numpy as np
import argparse
import shutil
from pathlib import Path
from mohou.model import LSTM
import torch
from hashlib import md5
import json
from typing import Any, Optional, TypeVar, Generic, List, Tuple
from dataclasses import asdict, dataclass, fields
from mohou.model import LSTM, LSTMConfig
from mohou.model.common import ModelBase
import torch
from mohou.types import EpisodeBundle
from mohou.trainer import TrainCache, TrainConfig
import uuid
from mohou.model.autoencoder import VariationalAutoEncoder, AutoEncoderConfig
from mohou.model.lstm import LSTM, LSTMConfig
from mohou.script_utils import train_autoencoder, train_lstm
from mohou.types import RGBImage, AngleVector
from mohou.default import create_default_encoding_rule, create_default_propagator
from mohou.dataset.sequence_dataset import AutoRegressiveDataset, AutoRegressiveDatasetConfig
from pathlib import Path
import argparse
from pathlib import Path
from typing import Optional, Type

from mohou.dataset import AutoEncoderDatasetConfig
from mohou.file import get_project_path
from mohou.model import AutoEncoder, AutoEncoderBase, VariationalAutoEncoder
from mohou.model.autoencoder import AutoEncoderConfig
from mohou.script_utils import create_default_logger, train_autoencoder
from mohou.setting import setting
from mohou.trainer import TrainConfig
from mohou.types import EpisodeBundle, ImageBase, RGBImage, get_element_type

import torch
from mohou.types import EpisodeBundle
from mohou.trainer import TrainCache
from mohou.model.autoencoder import VariationalAutoEncoder
from mohou.model.lstm import LSTM
from mohou.types import RGBImage, AngleVector
from mohou.default import create_default_encoding_rule, create_default_propagator
from mohou.dataset.sequence_dataset import AutoRegressiveDataset, AutoRegressiveDatasetConfig
from pathlib import Path


@dataclass
class ModelDigest:

    @dataclass
    class ParamDigest:
        mean: float
        std: float
        minimum: float
        maximum: float
        # DO NOT USE SUM

        def is_close_enough(self, other: "ModelDigest.ParamDigest", tol: float):
            for field in fields(self):
                val_self = self.__dict__[field.name]
                val_other = self.__dict__[field.name]
                if abs(val_self - val_other) > tol:
                    return False
            return True

    structure_hash: str
    param_specs: List[ParamDigest]

    @classmethod
    def from_model(cls, model: ModelBase):
        structure_hash = md5(str(model).encode('utf-8')).hexdigest()

        param_spec_list = []
        for param in model.parameters():
            spec = cls.ParamDigest(
                float(torch.mean(param).item()),
                float(torch.std(param).item()),
                float(torch.max(param).item()),
                float(torch.min(param).item()),
                )
            param_spec_list.append(spec)

        return cls(structure_hash, param_spec_list)

    def is_comparable(self, other: "ModelDigest") -> bool:
        return self.structure_hash == other.structure_hash

    def is_close_enough(self, other: "ModelDigest", tol = 1e-6) -> bool:
        # NOTE(to future-self): do not replace this function by md5sum + pickle.

        assert self.is_comparable(other)

        for spec, spec_other in zip(self.param_specs, other.param_specs):
            if not spec.is_close_enough(spec_other, tol):
                return False
        return True

    def dumps(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @classmethod
    def loads(cls, string: str) -> "ModelDigest":
        d = json.loads(string)
        param_specs = []
        for param_spec_dict in d["param_specs"]:
            param_specs.append(cls.ParamDigest(**param_spec_dict))
        d["param_specs"] = param_specs
        return cls(**d)

    @property
    def param_spec_hash(self) -> str:
        val = md5(str(self.param_specs).encode('utf-8')).hexdigest()
        return val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelDigest):
            return NotImplemented
        assert type(self) == type(other)
        return self.param_spec_hash == other.param_spec_hash


def dump_model_digest(model: ModelBase, debug_path: Path, postfix: Optional[str] = None):
    digest_path = debug_path / "digest"
    digest_path.mkdir(exist_ok=True)

    digest = ModelDigest.from_model(model)
    model_type_name = model.__class__.__name__
    uuid_value = uuid.uuid4().hex[:6]

    if postfix is None:
        postfix = uuid_value
    else:
        postfix = "{}-{}".format(postfix, uuid_value)

    file_path = digest_path / "digest-{}-{}".format(model_type_name, postfix)
    assert not file_path.exists()
    with file_path.open(mode = "w") as f:
        f.write(digest.dumps())

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-post", type=str, help="postfix")
    parser.add_argument("--warm", action="store_true", help="warm start")
    args = parser.parse_args()
    postfix: Optional[str] = args.post

    debug_path = Path("~/debug").expanduser() 
    debug_path.mkdir(exist_ok=True)

    #shutil.rmtree(debug_path / "mohou_data", ignore_errors=True)
    #cmd = "cd {} && git clone https://github.com/HiroIshida/mohou_data.git --depth 1".format(debug_path)
    #subprocess.run(cmd, shell=True)

    kuka_script_path = Path("kuka_reaching.py").expanduser()
    project_path = debug_path / "dummy_project"
    shutil.rmtree(project_path, ignore_errors=True)
    project_path.mkdir(exist_ok=True)
    subprocess.run("cp {} {}".format(Path("~/EpisodeBundle.tar"), project_path), shell=True)

    #subprocess.run("python3 kuka_reaching.py -pp {} -n 12 -m 28 -untouch 2".format(project_path), shell=True)

    vae_type = VariationalAutoEncoder
    image_type = RGBImage

    bundle = EpisodeBundle.load(project_path)
    bundle_md5sum = md5(pickle.dumps(bundle)).hexdigest()
    print("bundle md5sum => {}".format(bundle_md5sum))

    sumval = 0.0
    for episode in bundle:
        av_seq = episode.get_sequence_by_type(AngleVector)
        sumval += np.mean([av.numpy() for av in av_seq])
    print("bundle sumval => {}".format(sumval))

    print("start training autoencoder...")
    ae_config = AutoEncoderConfig(image_type, n_pixel=28)
    dataset_config = AutoEncoderDatasetConfig(0)
    train_config = TrainConfig(batch_size=10, n_epoch=10)
    train_autoencoder(project_path, image_type, ae_config, dataset_config, train_config, vae_type)
    ae_model = TrainCache.load(project_path, vae_type).best_model
    dump_model_digest(ae_model, debug_path, postfix)

    print("start training lstm...")
    rule = create_default_encoding_rule(project_path)
    lstm_config = LSTMConfig(rule.dimension, n_layer=1)
    dataset_config = AutoRegressiveDatasetConfig(n_aug=2)
    train_config = TrainConfig(batch_size=10, n_epoch=30)
    train_lstm(project_path, rule, lstm_config, dataset_config, train_config)
    lstm_model = TrainCache.load(project_path, LSTM).best_model
    dump_model_digest(lstm_model, debug_path, postfix)
