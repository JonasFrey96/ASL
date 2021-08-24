import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning import metrics as pl_metrics


class RandomDataset(Dataset):
  def __init__(self, size, length):
    self.len = length
    self.data = torch.randn(length, size)

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return self.len


class BoringModel(LightningModule):
  def __init__(self):
    super().__init__()
    self.layer = torch.nn.Linear(32, 2)
    self.train_acc = pl_metrics.classification.Accuracy(compute_on_step=False)

  def forward(self, x):
    return self.layer(x)

  def training_step(self, batch, batch_idx):
    print("total:", self.train_acc.correct)
    loss = self(batch).sum()
    self.log("train_loss", loss)

    # print(batch)
    self.train_acc.update(
      torch.randint(0, 10, batch.shape), torch.randint(0, 10, batch.shape)
    )

    self.log("ACC", self.train_acc, on_epoch=True, on_step=False)
    return {"loss": loss}

  def on_train_epoch_end(self, outputs):

    print("ACC", self.train_acc.compute())
    self.train_acc.reset()

  def validation_step(self, batch, batch_idx):
    loss = self(batch).sum()
    self.log("valid_loss", loss)

  def test_step(self, batch, batch_idx):
    loss = self(batch).sum()
    self.log("test_loss", loss)

  def configure_optimizers(self):
    return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
  train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
  val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
  test_data = DataLoader(RandomDataset(32, 64), batch_size=2)

  model = BoringModel()
  trainer = Trainer(
    default_root_dir=os.getcwd(),
    limit_train_batches=1.0,
    limit_val_batches=1.0,
    num_sanity_val_steps=0,
    max_epochs=5,
  )
  trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)


if __name__ == "__main__":
  run()
