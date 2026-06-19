import torch


def collate_fn(dataset_items):
    result_batch = {}

    for key in dataset_items[0].keys():
        values = [item[key] for item in dataset_items]

        if key == "image_id":
            result_batch[key] = values
        else:
            result_batch[key] = torch.stack(values)

    return result_batch
