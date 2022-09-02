import torch
from hashlib import md5
from typing import Any, Optional, TypeVar, Generic, List, Tuple
from dataclasses import dataclass, fields
from mohou.model import LSTM, LSTMConfig
from mohou.model.common import ModelBase

@dataclass
class ModelAbbreviation:

    @dataclass
    class ParamSpec:
        mean: float
        std: float
        minimum: float
        maximum: float
        # DO NOT USE SUM

        def is_close_enough(self, other: "ModelAbbreviation.ParamSpec", tol: float):
            for field in fields(self):
                val_self = self.__dict__[field.name]
                val_other = self.__dict__[field.name]
                if abs(val_self - val_other) > tol:
                    return False
            return True

    structure_hash: str
    param_specs: List[ParamSpec]

    @classmethod
    def from_model(cls, model: ModelBase):
        structure_hash = md5(str(model).encode('utf-8')).hexdigest()

        param_spec_list = []
        for param in model.parameters():
            spec = cls.ParamSpec(
                float(torch.mean(param).item()),
                float(torch.std(param).item()),
                float(torch.max(param).item()),
                float(torch.min(param).item()),
                )
            param_spec_list.append(spec)
        return cls(structure_hash, param_spec_list)

    def is_comparable(self, other: "ModelAbbreviation") -> bool:
        return self.structure_hash == other.structure_hash

    def is_close_enough(self, other: "ModelAbbreviation", tol = 1e-6) -> bool:
        # NOTE(to future-self): do not replace this function by md5sum + pickle.

        assert self.is_comparable(other)

        for spec, spec_other in zip(self.param_specs, other.param_specs):
            if not spec.is_close_enough(spec_other, tol):
                return False
        return True



print(str(LSTMConfig(10)))
model1 = LSTM(LSTMConfig(10))
model2 = LSTM(LSTMConfig(10))
