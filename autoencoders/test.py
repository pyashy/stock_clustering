from autoencoders import Conv1dAutoEncoder, TickerDataModule

from pytorch_lightning import Trainer

if __name__ == '__main__':
    model = Conv1dAutoEncoder(1, 1)
    trainer = Trainer(gpus=1, max_epochs=100)

    dm = TickerDataModule('..\\data\\ticker_data_Close.csv')

    trainer.fit(model, dm)





