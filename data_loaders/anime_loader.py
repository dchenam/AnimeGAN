import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class Anime_Dataset(Dataset):
    def __init__(self, config, transform):
        self.config = config
        self.transform = transform
        self.lines = open(config.label_path, 'r').readlines()
        self.num_data = len(self.lines)
        self.image_ids = []
        self.labels = []
        self.tag_dict = {'orange_hair': 0, 'white_hair': 1, 'aqua_hair': 2, 'gray_hair': 3, 'green_hair': 4,
                         'red_hair': 5, 'purple_hair': 6, 'pink_hair': 7, 'blue_hair': 8, 'black_hair': 9,
                         'brown_hair': 10, 'blonde_hair': 11, 'gray_eyes': 12, 'black_eyes': 13, 'orange_eyes': 14,
                         'pink_eyes': 15, 'yellow_eyes': 16, 'aqua_eyes': 17, 'purple_eyes': 18, 'green_eyes': 19,
                         'brown_eyes': 20, 'red_eyes': 21, 'blue_eyes': 22, 'bicolored_eyes': 23}
        print('preprocessing...')
        print('number of images: ', self.num_data)
        self.preprocess()

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        correct_image = Image.open(os.path.join(self.config.image_dir, self.image_ids[index] + '.jpg'))
        correct_text = self.labels[index]
        # wrong_text = self.labels[np.random.randint(low=0, high=self.num_data)]
        random_index = np.random.randint(low=0, high=self.num_data)
        wrong_image = Image.open(os.path.join(self.config.image_dir, self.image_ids[random_index] + '.jpg'))
        return self.transform(correct_image), torch.Tensor(correct_text), self.transform(wrong_image)

    def preprocess(self):
        for i, line in enumerate(self.lines):
            splits = line.split()
            image_id = splits[0]
            attr_values = splits[1:]
            one_hot = np.zeros(len(self.tag_dict))
            for value in attr_values:
                index = self.tag_dict[value]
                one_hot[index] = 1
            self.labels += [one_hot]
            self.image_ids += [image_id]

    def generate_embedding(self):
        test_str = ['blue_hair, red_eyes', 'brown_hair, brown_eyes', 'black_hair, blue_eyes', 'red_hair, green_eyes']
        embeddings = {}
        for str in test_str:
            split = str.split(', ')
            one_hot = np.zeros(len(self.tag_dict))
            for tag in split:
                one_hot[self.tag_dict[tag]] = 1
            embeddings[str] = one_hot
        return embeddings


def get_loader(config):
    transform = transforms.Compose([
        # transforms.CenterCrop(config.crop_size),
        transforms.Scale(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),  # 3 for RGB channels
                             std=(0.5, 0.5, 0.5))
    ])

    dataset = Anime_Dataset(config, transform)

    print('generating test embeddings...')
    embeddings = dataset.generate_embedding()

    data_loader = DataLoader(dataset,
                             config.batch_size,
                             shuffle=True,
                             num_workers=4,
                             drop_last=True)
    return data_loader, embeddings
