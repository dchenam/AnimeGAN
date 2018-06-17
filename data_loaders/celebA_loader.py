import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image


class Celeb_Dataset(Dataset):
    def __init__(self, config, transform):
        self.config = config
        self.transform = transform
        self.lines = open(config.metafile_path, 'r').readlines()
        self.num_data = self.lines[0]
        print('preprocessing...')
        self.preprocess()

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.config.image_dir, self.filenames[index]))
        return self.transform(image)

    def preprocess(self):
        self.filenames = []
        img_lines = self.lines[2:]
        random.shuffle(img_lines)
        for i, line in enumerate(img_lines):
            splits = line.split()
            filename = splits[0]
            attr_values = splits[1:]
            self.filenames += [filename]


def get_loader(config):
    if config.mode == 'train':
        transform = transforms.Compose([
            transforms.CenterCrop(config.crop_size),
            transforms.Scale(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    dataset = Celeb_Dataset(config, transform)

    data_loader = DataLoader(dataset,
                             config.batch_size,
                             shuffle=True,
                             num_workers=config.workers)
    return data_loader
