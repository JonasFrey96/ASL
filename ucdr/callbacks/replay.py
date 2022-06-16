# TODO: Write a test and verify replaycallback always acts strictly contracting.

from pytorch_lightning.callbacks import Callback
import torch

__all__ = ["ReplayCallback"]


class ReplayCallback(Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        if pl_module._exp["replay"]["cfg_rssb"]["elements"] != 0:
            ls_global_indices = trainer.train_dataloader.dataset.datasets.get_replay_datasets_globals()
            bins_np, valid_np = pl_module._rssb.get()

            for i, ls in enumerate(ls_global_indices):
                elements = (bins_np[i][valid_np[i]]).tolist()
                # ensure to only allow contraction of the global indices within the dataset!
                for e in elements:
                    assert e in ls
                ls_global_indices[i] = elements

            trainer.train_dataloader.dataset.datasets.set_replay_datasets_globals(ls_global_indices)

    def on_train_end(self, trainer, pl_module):
        if pl_module._exp["replay"]["cfg_rssb"]["elements"] != 0:
            ls_global_indices = trainer.train_dataloader.dataset.datasets.get_datasets_globals()
            assert len(ls_global_indices) <= pl_module._rssb.bins.shape[0]

            # PERFORMS RANDOM MEMORY BUFFER FILLING
            # IDEA IS WE CAN ITERATE OVER ALL BINS AND RANDOMLY SAMPLE
            # IIF. THE BIN IS NOT VALID JET
            bin_fill_status = pl_module._rssb.valid.sum(dim=1)

            for bin, ls in enumerate(ls_global_indices):
                if bin_fill_status[bin] == int(pl_module._rssb.limits[bin]):
                    # DONT FILL BECAUSE ALREADY VALID ENTRIES
                    continue

                el = int(pl_module._rssb.limits[bin])
                torch_ls = torch.tensor(ls, dtype=torch.long)

                if el >= len(ls):
                    # FOR DATASETS THAT ALREADY HAVE BEEN CONTRACTED NOTHING CHANGES.
                    pl_module._rssb.bins[bin, : len(ls)] = torch_ls[torch.arange(0, len(ls), dtype=torch.long)]
                    pl_module._rssb.valid[bin, : len(ls)] = True
                    pl_module._rssb.valid[bin, len(ls) :] = False
                else:
                    pl_module._rssb.bins[bin, :el] = torch_ls[torch.randperm(len(ls))[:el]]
                    pl_module._rssb.valid[bin, :el] = True
                val = min(el, 5)
                print(f"Set for bin {bin} following indices: ", pl_module._rssb.bins[bin, :val])
