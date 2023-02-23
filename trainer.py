import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class Trainer():
    def __init__(self, data_source, preprocessor, model, **kwargs) -> None:
        self.data_source = data_source
        self.preprocessor = preprocessor
        self.model = model

        self.log_metrics = {}
        self.log_params = {}
        self.log_models = []

        self.data = None
        self.train_data = None
        self.valid_data = None

        if "shuffle" in kwargs:
            self.shuffle = kwargs["shuffle"]
        else:
            self.shuffle = False

        if "lr" in kwargs:
            self.lr = kwargs["lr"]
        else:
            self.lr = 1e-4

        if "epochs" in kwargs:
            self.epochs = kwargs["epochs"]
        else:
            self.epochs = 100
        
        if "test_size" in kwargs:
            self.test_size = kwargs["test_size"]
        else:
            self.test_size = 0.2

        if "batch_size" in kwargs:
            self.batch_size = kwargs["batch_size"]
        else:
            self.batch_size = 20

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.loss_fucntion = torch.nn.functional.mse_loss

        self.training_loss = []
        self.validation_loss = []

        self.load_data()
        self.get_train_test_data()

        pass

    def train(self):
        for epoch in range(self.epochs):
            self.valid_one_epochs()
            self.train_one_epochs()
            if (epoch+1)%5==0 or epoch==1:
                print(f"[{epoch+1}/{self.epochs}], training_loss:{self.training_loss[-1]}, validation_loss:{self.validation_loss[-1]}")
        return None

    def train_one_epochs(self):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (x,y) in enumerate(self.train_data):
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_fucntion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        self.training_loss.append(round(running_loss/(batch_idx+1),5))
        return None

    def valid_one_epochs(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x,y) in enumerate(self.valid_data):
                y_pred = self.model(x)
                loss = self.loss_fucntion(y_pred, y)
                running_loss += loss.item()
        self.validation_loss.append(round(running_loss/(batch_idx+1),5))
        return None

    def load_data(self):
        from pathlib import Path
        import pandas as pd
        data = [pd.read_csv(f,encoding="utf_8_sig") for f in Path(self.data_source).glob("*.csv")]
        self.data = pd.concat(data,axis=0).values[:,1:7]
        return None

    def _make_data_loader(self, dataset, batch_size=1):
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
        return dataloader

    def get_train_test_data(self):
        train_data, valid_data = train_test_split(self.data, test_size=self.test_size)
        dataset_train = MakeDataset(train_data)
        dataset_valid = MakeDataset(valid_data)
        self.train_data = self._make_data_loader(dataset_train, batch_size=self.batch_size)
        self.valid_data = self._make_data_loader(dataset_valid, batch_size=20)
        return None

    def export_log_model(self):
        log_models = [("model",self.model,"pytorch")]
        return log_models

class MakeDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.x = torch.Tensor(data[:,:5]).unsqueeze(dim=1)
        self.y = torch.Tensor(data[:,5:6])
        self.len = len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

