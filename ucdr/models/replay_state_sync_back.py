import torch
import torch.nn as nn

# This model is used to store the state of the replay buffer in the dataloaders.
# This allows to easly continue training with the correct buffer mode from a checkpoint.
# Simply at the begining of each epoch write the buffer state to the dataloader shared memory.
# And at the end of the epoch copy the shared memory to the model.


__all__ = ["ReplayStateSyncBack"]


class ReplayStateSyncBack(nn.Module):
    def __init__(self, bins, elements, percantage=5, use_percantage=False, dataset_sizes=[]):
        super().__init__()

        if use_percantage:
            # create an array with the largest buffer necessary
            self.limits = torch.tensor(dataset_sizes) * (percantage / 100)
            elements = int(self.limits.max())
        else:
            self.limits = torch.tensor([elements] * bins)
        self.limits.type(torch.long)
        self.register_buffer("bins", torch.zeros((bins, elements), dtype=torch.long), persistent=True)
        self.register_buffer("valid", torch.zeros((bins, elements), dtype=torch.bool), persistent=True)
        self.nr_elements = elements
        self.nr_bins = bins
        self.use_percantage = use_percantage

    def absorbe(self, bins, valid):
        self.bins = torch.from_numpy(bins).type(torch.long)
        self.valid = torch.from_numpy(valid).type(torch.bool)
        if self.use_percantage:
            for i in range(self.limits.shape[0]):
                if self.bins[i, : self.limits[i]].sum() != 0:
                    raise Exception("RSSB: Elements added with invalid index")
                if self.valids[i, : self.limits[i]].sum() != 0:
                    raise Exception("RSSB: Elements added with invalid index")

                self.bins[i, : self.limits[i]] = 0
                self.valids[i, : self.limits[i]] = False

    def get(self):
        return self.bins.to("cpu").numpy(), self.valid.to("cpu").numpy()
