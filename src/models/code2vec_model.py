from typing import Any, Dict, Optional

from encoders import NBoWEncoder, Code2VecEncoder
from .code2vec_model_base import Code2VecModelBase


class Code2VecModel(Code2VecModelBase):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        hypers = {}
        hypers.update({f'code_{key}': value for key, value in Code2VecEncoder.get_default_hyperparameters().items()})
        hypers.update({f'query_{key}': value for key, value in NBoWEncoder.get_default_hyperparameters().items()})
        model_hypers = {
            'code_use_subtokens': False,
            'code_mark_subtoken_end': False,
            'loss': 'triplet',
            'batch_size': 200
        }
        hypers.update(super().get_default_hyperparameters())
        hypers.update(model_hypers)
        return hypers

    def __init__(self,
                 hyperparameters: Dict[str, Any],
                 run_name: str = None,
                 model_save_dir: Optional[str] = None,
                 log_save_dir: Optional[str] = None):
        super().__init__(
            hyperparameters,
            code_encoder_type=Code2VecEncoder,
            query_encoder_type=NBoWEncoder,
            run_name=run_name,
            model_save_dir=model_save_dir,
            log_save_dir=log_save_dir)