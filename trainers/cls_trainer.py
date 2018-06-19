from base.base_train import BaseTrain
from tqdm import trange
from torchvision.utils import save_image

import os
import torch
import torch.nn as nn

class CLSTrainer(BaseTrain):
    def __init__(self, G, D, config, data, embedding, logger, device):
        super(CLSTrainer, self).__init__(G, D, config, data, logger, device)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=config.learning_rate,
                                            betas=(config.beta1, config.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=config.learning_rate,
                                            betas=(config.beta1, config.beta2))
        self.criterion = nn.BCELoss()
        self.data_iter = iter(self.data)
        self.embedding = embedding
        self.g_loss = 0
        self.d_loss = 0

    def train(self):
        for train_iter in trange(self.config.num_iters):
            self.train_step(train_iter)
            if train_iter % self.config.sample_iters == 0:
                image = self.sample().data.cpu()
                sample_path = os.path.join(self.config.sample_dir, '{}-sample.jpg'.format(train_iter))
                save_image(self.denorm(image), sample_path, nrow=2, padding=2)
            if train_iter % self.config.save_iters == 0:
                self.save_models(train_iter)

    def train_step(self, step):
        ## Pre-Process Input
        try:  # Iter Stopping Problem
            real_image, real_text, wrong_image = next(self.data_iter)
        except:
            self.data_iter = iter(self.data)
            real_image, real_text, wrong_image = next(self.data_iter)

        real_image = real_image.to(self.device)
        real_text = real_text.to(self.device)
        wrong_image = wrong_image.to(self.device)

        # Create the labels which are later used as input for the BCE loss
        real_label = torch.ones(self.config.batch_size, 1).to(self.device)
        fake_label = torch.zeros(self.config.batch_size, 1).to(self.device)

        m = torch.distributions.Normal(0, 1)
        z = m.sample((self.config.batch_size, self.config.z_dim)).to(self.device)

        """Train Discriminator"""
        fake_image = self.G(z, real_text)

        real_loss = torch.mean(self.criterion(self.D(real_image, real_text), real_label))
        wrong_loss = torch.mean(self.criterion(self.D(wrong_image, real_text), fake_label))
        fake_loss = torch.mean(
            self.criterion(self.D(fake_image.detach(), real_text), fake_label))  # stop gradient into G
        self.d_loss = real_loss + (wrong_loss + fake_loss) / 2

        self.d_optimizer.zero_grad()
        self.d_loss.backward()
        self.d_optimizer.step()

        """Train Generator"""
        fake_image = self.G(z, real_text)
        self.g_loss = torch.mean(self.criterion(self.D(fake_image, real_text), real_label))
        self.g_optimizer.zero_grad()
        self.g_loss.backward()
        self.g_optimizer.step()

        self.logger.scalar_summary("fake_loss", fake_loss, step)
        self.logger.scalar_summary("wrong_loss", wrong_loss, step)
        self.logger.scalar_summary("real_loss", real_loss, step)
        self.logger.scalar_summary("d_loss", self.d_loss, step)
        self.logger.scalar_summary("g_loss", self.g_loss, step)

    def sample(self):
        sample_strs = ['blue_hair, red_eyes', 'brown_hair, brown_eyes', 'black_hair, blue_eyes', 'red_hair, green_eyes']
        sample_embeddings = [self.embedding[str] for str in sample_strs]
        with torch.no_grad():
            m = torch.distributions.Normal(0, 0.3)
            z = m.sample((len(sample_strs), self.config.z_dim)).to(self.device)
            sample_text = torch.Tensor(sample_embeddings).to(self.device)
            out = torch.squeeze(self.G(z, sample_text))
        return out

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def save_models(self, train_iter):
        torch.save(self.G.state_dict(), os.path.join(self.config.checkpoint_dir, '{}-G.ckpt'.format(train_iter)))
        torch.save(self.D.state_dict(), os.path.join(self.config.checkpoint_dir, '{}-D.ckpt'.format(train_iter)))

    def load_models(self, resume_iter):
        self.G.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, '{}-G.pkl'.format(resume_iter))))
        self.D.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir, '{}-D.pkl'.format(resume_iter))))
