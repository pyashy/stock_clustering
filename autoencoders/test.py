from autoencoders import Conv1dAutoEncoder, LSTMAutoEncoder, TickerDataModule

from pytorch_lightning import Trainer


def test_conv_encoder():
    model = Conv1dAutoEncoder(1, 1)
    trainer = Trainer(gpus=1, max_epochs=100)

    dm = TickerDataModule('..\\data\\ticker_data_Close.csv')

    trainer.fit(model, dm)


def test_lstm_encoder():
    model = LSTMAutoEncoder(30, 1)
    trainer = Trainer(gpus=1, max_epochs=100)

    dm = TickerDataModule('..\\data\\ticker_data_Close.csv', batch_size=1)

    trainer.fit(model, dm)


if __name__ == '__main__':
    # test_conv_encoder()
    test_lstm_encoder()





