class BaseTrain:
    def __init__(self, G, D, config, data, logger, device):
        self.G = G
        self.D = D
        self.config = config
        self.data = data
        self.logger = logger
        self.device = device

    def train(self):
        for cur_epoch in range(self.config.num_epochs):
            self.train_epoch()

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self, iter):
        """
        implement the logic of the train step
        - return any metrics you need to summarize
        """
        raise NotImplementedError
