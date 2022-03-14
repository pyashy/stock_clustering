from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import time
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT, EPOCH_OUTPUT


def init_weights(m):
    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
    m.bias.data.fill_(0.01)


class LSTMAutoEncoder(LightningModule):

    def __init__(self, seq_len, n_features, embedding_dim=16):
        super(LSTMAutoEncoder, self).__init__()

        self.encoder1 = nn.LSTM(input_size=n_features,
                                hidden_size=embedding_dim * 2,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.3)
        self.encoder2 = nn.LSTM(input_size=embedding_dim * 2,
                                hidden_size=embedding_dim,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.3)

        self.decoder1 = nn.LSTM(input_size=embedding_dim,
                                hidden_size=embedding_dim,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.3)
        self.decoder2 = nn.LSTM(input_size=embedding_dim,
                                hidden_size=embedding_dim * 2,
                                num_layers=1,
                                batch_first=True,
                                dropout=0.3)

        # self.encoder = nn.Sequential(
        #     nn.LSTM(input_size=n_features, hidden_size=embedding_dim * 2, num_layers=1, batch_first=True),
        #     nn.LSTM(input_size=embedding_dim * 2, hidden_size=embedding_dim, num_layers=1, batch_first=True),
        # )

        # self.decoder = nn.Sequential(
        #     nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True),
        #     nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim * 2, num_layers=1, batch_first=True)
        # )

        self.output_layer = nn.Linear(embedding_dim * 2, n_features)

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.time_start = time.time()

    # TODO РА
    def forward(self, x: Tensor, *args, **kwargs) -> Any:
        x = x.reshape((1, self.seq_len, self.n_features))
        # print(x.shape)
        x, _ = self.encoder1(x)
        x, (hidden_n, _) = self.encoder2(x)

        h = hidden_n.reshape((self.n_features, self.embedding_dim))
        h = h.repeat(self.seq_len, self.n_features)
        h = h.reshape((self.n_features, self.seq_len, self.embedding_dim))

        h, _ = self.decoder1(h)
        h, _ = self.decoder2(h)
        h = h.reshape((self.seq_len, self.embedding_dim * 2))

        return self.output_layer(h)

    def training_step(self, batch: Tensor, *args, **kwargs) -> STEP_OUTPUT:
        batch_rec = self(batch)
        loss = torch.nn.MSELoss()(batch_rec, batch)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log('train_time', time.time() - self.time_start, prog_bar=True)

    def validation_step(self, batch: Tensor, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        batch_rec = self(batch)
        loss = torch.nn.MSELoss()(batch_rec, batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def test_step(self, batch: Tensor, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        batch_rec = self(batch)
        loss = torch.nn.MSELoss()(batch_rec, batch)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

        def adjust_lr(epoch):
            if epoch < 100:
                return 0.003
            if 100 <= epoch < 120:
                return 0.0003
            if 120 <= epoch < 150:
                return 0.000003
            if 150 <= epoch < 200:
                return 0.0000003
            else:
                return 0.00000003

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=adjust_lr
            ),
            "name": "lr schedule",
        }
        return [optimizer], [lr_scheduler]