import argparse
from model.utils import EasyConfig
from model.ChexnetTrainer import ChexnetTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Weakly supervised lesion localization training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--img', type=str, required=False, help='img to generate heatmap',
                        default="F:\\dataset\\CXR14\\images\\00030323_015.png")
    parser.add_argument('--thr', type=str, required=False, help='threshhold to judge positive sample',
                        default=[0.4852481, 0.54961604, 0.55970126, 0.4372623, 0.56016654, 0.4980255, 0.17124416, 0.5572045, 0.34506968, 0.4473719, 0.6461501, 0.4906137, 0.40028474, 0.16027835])

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    if cfg.mode == 'train':
        ChexnetTrainer.train(cfg)
        ChexnetTrainer.test(cfg)
    elif cfg.mode == 'test':
        ChexnetTrainer.test(cfg)
    elif cfg.mode == 'heatmap':
        ChexnetTrainer.heatmap(args.img, args.thr, cfg)
    elif cfg.mode == 'localization':
        ChexnetTrainer.localization(0.5, cfg)
    else:
        raise ValueError(f"model {cfg.mode} is unknown")