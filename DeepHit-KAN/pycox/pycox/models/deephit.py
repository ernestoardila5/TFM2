import numpy as np
import pandas as pd
import torch
import torchtuples as tt

from pycox import models
from pycox.models.utils import pad_col

class DeepHitSingle(models.pmf.PMFBase): 
    def __init__(self, net, optimizer=None, device=None, duration_index=None, alpha=0.2, sigma=0.1, loss=None): #net es la red neuronal
        
        if loss is None:
            loss = models.loss.DeepHitSingleLoss(alpha, sigma)
        super().__init__(net, loss, optimizer, device, duration_index)

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=models.data.DeepHitDataset)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader


class DeepHit(tt.Model):
    def __init__(self, net, optimizer=None, device=None, alpha=0.2, sigma=0.1, duration_index=None, loss=None):
        self.duration_index = duration_index
        if loss is None:
            loss = models.loss.DeepHitLoss(alpha, sigma)
        super().__init__(net, loss, optimizer, device)

    @property
    def duration_index(self):
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=models.data.DeepHitDataset)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        return pd.DataFrame(surv, self.duration_index)

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        cif = self.predict_cif(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = 1. - cif.sum(0)
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_cif(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        pmf = self.predict_pmf(input, batch_size, False, eval_, to_cpu, num_workers)
        cif = pmf.cumsum(1)
        return tt.utils.array_or_tensor(cif, numpy, input)

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        pmf = pad_col(preds.view(preds.size(0), -1)).softmax(1)[:, :-1]
        pmf = pmf.view(preds.shape).transpose(0, 1).transpose(1, 2)
        return tt.utils.array_or_tensor(pmf, numpy, input)
