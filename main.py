import torch
from torch.backends import cudnn
from data_loaders.anime_loader import get_loader
from models.cls import Generator, Discriminator
from trainers.cls_trainer import CLSTrainer
from logger import Logger
from utils import *
import logging


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiment dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.sample_dir])

    # create the data loader
    data_loader, embedding = get_loader(config)

    # create tensorboard logger
    logger = Logger(config)
    logger.set_logger(log_path=os.path.join(config.model_dir, args.mode + '.log'))

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # create instance of the model
    G = Generator(config).to(device)
    D = Discriminator(config).to(device)

    # check parameters
    print_network(G, "generator")
    print_network(D, "discriminator")

    # create trainer
    trainer = CLSTrainer(G, D, config, data_loader, embedding, logger, device)
    # enter training or testing
    if args.mode == 'train':
        if args.resume:
            trainer.load_models(config.resume_epoch)
        # train the models
        trainer.train()
    else:
        trainer.load_models(config.test_epoch)
        trainer.sample()
    logging.info("finished...")


if __name__ == "__main__":
    main()
