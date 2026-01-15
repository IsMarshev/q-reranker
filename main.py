from schemas import load_config
from train_stage1 import Trainer

cfg = load_config('train_config.yml')

trainer = Trainer(cfg)
