from .attention_module.train_dl_model import main as _train_main


class Model:
    def run(self):
        _train_main()
