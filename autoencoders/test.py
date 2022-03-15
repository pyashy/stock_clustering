from autoencoders import Conv1dAutoEncoder, LSTMAutoEncoder, TickerDataModule, MLPAutoEncoder

from pytorch_lightning import Trainer
import pandas as pd


def test_conv_encoder():
    model = Conv1dAutoEncoder(1, 1)
    trainer = Trainer(gpus=1, max_epochs=100)

    dm = TickerDataModule('..\\data\\ticker_data_preprocessed.csv', preprocessing=False)

    trainer.fit(model, dm)
    trainer.test(model, dm)


def test_lstm_encoder():
    model = LSTMAutoEncoder(100, 1, embedding_dim=18)
    trainer = Trainer(gpus=1, max_epochs=100)

    dm = TickerDataModule('..\\data\\ticker_data_preprocessed.csv', batch_size=18, time_period=100, preprocessing=False)

    trainer.fit(model, dm)
    trainer.test(model, dm)


def test_mlp_encoder():
    model = MLPAutoEncoder(30, 8)
    trainer = Trainer(gpus=1, max_epochs=150)

    dm = TickerDataModule('..\\data\\ticker_data_preprocessed.csv', preprocessing=False)

    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == '__main__':

    test_lstm_encoder()
    # test_conv_encoder()
    # test_mlp_encoder()



