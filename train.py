from configparser import ConfigParser
import argparse
from dbm.gan_seq import *

def main():

    #torch device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU will be used!")
    else:
        device = torch.device("cpu")
        print("Couldn't find GPU, will use CPU instead!")
    torch.set_num_threads(12)

    #Read config
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    config_file = args.config

    cfg = ConfigParser(inline_comment_prefixes="#")
    cfg.read(config_file)

    #set up model
    model = GAN_seq(device=device, cfg=cfg)

    #train
    model.train()

if __name__ == "__main__":
    main()
