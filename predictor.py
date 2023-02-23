import numpy as np
import torch

class Predictor():
    def __init__(self,**kwargs) -> None:
        self.model = kwargs["model"]
        self.model.eval()
        pass

    def predict(self, x:np.ndarray):
        x = torch.Tensor(x)
        x = x.unsqueeze(dim=1)
        with torch.no_grad():
            y = self.model(x)
        y = y.numpy()
        return y[:,0]