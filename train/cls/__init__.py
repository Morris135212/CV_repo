import torch
import torchvision.datasets as datasets
from promise.dataloader import DataLoader
from torch.utils.data import random_split

from model import weights_init


class Trainer:
    def __init__(self,
                 model,
                 train_data,
                 val_data,
                 initialize=True,
                 batch_size=32,
                 epochs=10,
                 optimizer="adam",
                 lr=1e-2,
                 momentum=0.5,
                 step_size=20,
                 interval=1,
                 patience=20,
                 # include_weight=True,
                 path="output/checkpoints/checkpoint.pt"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size)
        self.batch_size=batch_size
        self.epochs = epochs

        if initialize:
            model.apply(weights_init)

        self.model = model.to(device=self.device)
        # Optimizer
        if optimizer == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=step_size, gamma=0.5)
        self.interval = interval
        # self.include_weight = include_weight
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)
