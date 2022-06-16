import torch

__all__ = ["cross_entropy_soft"]


def cross_entropy_soft(pred, target, mask):
    """

    Parameters
    ----------
    pred : torch.tensor
        [BS,C,H,W]
    target : torch.tensor
         [BS,C,H,W] valid probability dist. C must sum up to 1!
    Returns
    -------
    loss : [torch.tensor]
        [mean soft label cross entropy per sample]
    """
    assert mask.sum() != 0

    pred = torch.nn.functional.log_softmax(pred, dim=1)

    res = -(target * pred).sum(dim=1)
    res = (res * mask).mean(dim=[1, 2]).mean()

    return res


def test():
    BS, C, H, W = 8, 40, 500, 520

    # create hard label
    label_ori = torch.randint(low=0, high=C - 1, size=(BS, H, W))

    label = torch.nn.functional.one_hot(label_ori.clone(), num_classes=C).permute(0, 3, 1, 2)

    label_soft = torch.rand((BS, C, H, W))
    label_soft = label_soft / torch.sum(label_soft, dim=1)[:, None].repeat(1, C, 1, 1)

    pred = torch.rand((BS, C, H, W))
    pred = pred / torch.sum(pred, dim=1)[:, None].repeat(1, C, 1, 1)

    res1 = cross_entropy_soft(pred, label)
    res2 = torch.nn.functional.cross_entropy(pred, label_ori, ignore_index=-1, reduction="none").mean(dim=[1, 2])
    if res1.mean() != res2.mean():
        raise Exception()

    _ = cross_entropy_soft(pred, label_soft)


if __name__ == "__main__":
    test()
