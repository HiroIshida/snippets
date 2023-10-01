from mohou.types import VectorBase, AngleVector, ElementDict, EpisodeData, ElementSequence, EpisodeBundle
from mohou.dataset import AutoRegressiveDatasetConfig
from mohou.file import get_project_path, create_project_dir
from mohou.script_utils import create_default_logger, train_lstm
from mohou.encoder import VectorIdenticalEncoder 
from mohou.encoding_rule import EncodingRule
from dataset import TargetPosition
from mohou.types import AngleVector, TerminateFlag
from mohou.trainer import TrainConfig
from mohou.model.lstm import PBLSTM, PBLSTMConfig


project_path =  get_project_path("pblstm_test")
bundle = EpisodeBundle.load(project_path)
encoding_rule = EncodingRule.from_encoders(
        [
            VectorIdenticalEncoder.create(TargetPosition, 3),
            VectorIdenticalEncoder.create(AngleVector, 7),
            VectorIdenticalEncoder.create(TerminateFlag, 1)
        ], bundle = bundle)


logger = create_default_logger(project_path, "pblstm")  # noqa
dataset_config = AutoRegressiveDatasetConfig(n_aug=10, cov_scale=0.1)
train_config = TrainConfig(n_epoch=2000)

model_config = PBLSTMConfig(
    encoding_rule.dimension,
    n_hidden=200,
    n_layer=2,
    n_pb_dim=1,
    n_pb=len(bundle),
)

train_lstm(
    project_path,
    encoding_rule,
    model_config,
    dataset_config,
    train_config,
    model_type=PBLSTM,
)
