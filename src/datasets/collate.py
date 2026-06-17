import torch


def collate_fn(dataset_items):
    result_batch = {}

    for key in dataset_items[0].keys():
        result_batch[key] = torch.stack([item[key] for item in dataset_items])

    return result_batch
