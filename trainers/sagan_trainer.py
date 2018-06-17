from base.base_train import BaseTrain
import os
from tqdm import trange
import torch


class SAGANTrainer(BaseTrain):
    def __init__(self, G, D, data, config, logger):
        super(SAGANTrainer, self).__init__(G, D, data, config, logger)
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=config.learning_rate,
                                            betas=(config.beta1, config.beta2))
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=config.learning_rate,
                                            betas=(config.beta1, config.beta2))

    def train(self):
        for epoch in self.config.num_epochs:
            self.train_epoch()

    def train_epoch(self):
        for it in trange(self.config.iter_per_epoch):
            self.train_step()

    def train_step(self):
        pass

    def sample(self):
        z = torch.rand(self.config.batch_size, self.config.z_dim)
        out = torch.squeeze(self.G(z))
        return out

    def save_models(self):
        torch.save(self.G.state_dict(), os.path.join(self.config.checkpoint_dir, self.config.exp_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.config.checkpoint_dir, self.config.exp_name + '_D.pkl'))

    def load_models(self):
        self.G.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, self.config.exp_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, self.config.exp_name + '_D.pkl')))
