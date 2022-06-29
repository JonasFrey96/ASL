import torch
from torchvision import transforms
from math import ceil

from ucdr.datasets import get_dataset

__all__ = ["eval_lists_into_dataloaders", "get_dataloader_test", "get_dataloader_train"]


def eval_lists_into_dataloaders(eval_lists, env, exp):
    loaders = []
    for eval_task in eval_lists:

        loaders.append(get_dataloader_test(eval_task.dataset_test_cfg, env, exp))
    return loaders


def get_dataloader_test(d_test, env, exp):
    output_transform = transforms.Compose(
        [
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # dataset and dataloader
    dataset_test = get_dataset(
        **d_test,
        env=env,
        output_trafo=output_transform,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        num_workers=max(1, ceil(exp["loader"]["num_workers"] / torch.cuda.device_count())),
        pin_memory=exp["loader"]["pin_memory"],
        batch_size=exp["loader"]["batch_size"],
        drop_last=False,
    )
    return dataloader_test


def get_dataloader_train(d_train, env, exp):
    print("Number CUDA Devices: " + str(torch.cuda.device_count()))

    output_transform = transforms.Compose(
        [
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # dataset and dataloader
    dataset_train = get_dataset(
        **d_train,
        env=env,
        output_trafo=output_transform,
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=exp["loader"]["shuffle"],
        num_workers=ceil(exp["loader"]["num_workers"] / torch.cuda.device_count()),
        pin_memory=exp["loader"]["pin_memory"],
        batch_size=exp["loader"]["batch_size"],
        drop_last=True,
    )

    # dataset and dataloader
    dataset_buffer = get_dataset(
        **d_train,
        env=env,
        output_trafo=output_transform,
    )
    dataset_buffer.replay = False
    dataset_buffer.unique = True

    dataloader_buffer = torch.utils.data.DataLoader(
        dataset_buffer,
        shuffle=False,
        num_workers=ceil(exp["loader"]["num_workers"] / torch.cuda.device_count()),
        pin_memory=exp["loader"]["pin_memory"],
        batch_size=max(1, ceil(exp["loader"]["batch_size"])),
        drop_last=True,
    )

    return dataloader_train, dataloader_buffer
