from .attention_module.train_dl_model import main as _train_attention
from .stacking.train_stacking import main as _train_stacking


class Model:
    def run(self):
        _train_attention()
        _train_stacking()
