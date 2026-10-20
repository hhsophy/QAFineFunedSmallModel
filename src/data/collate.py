import torch
from functools import partial
from torch.utils.data import Sampler

class ResumableRandomSampler(Sampler):
    def __init__(self, dataset, seed=123):
        self.dataset = dataset
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = None
        self.position = 0

    def __iter__(self):
        if (
            self.permutation is None
            or self.position >= len(self.permutation)
        ):
            self.permutation = torch.randperm(
                len(self.dataset),
                generator=self.generator
            ).tolist()
            self.position = 0

        while self.position < len(self.permutation):
            idx = self.permutation[self.position]
            self.position += 1
            yield idx

    def __len__(self):
        return len(self.dataset)

    def state_dict(self):
        return {
            "permutation": self.permutation,
            "position": self.position,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        self.permutation = state["permutation"]
        self.position = state["position"]
        generator_state = torch.as_tensor(
        state["generator_state"],
        dtype=torch.uint8,
        device="cpu"
                )

        self.generator.set_state(generator_state)
    def reset(self):
        self.permutation = None
        self.position = 0

def custom_collate_fn(batch,pad_token_id=50256,ignore_index=-100,allowed_max_length=None,device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (new_item + [pad_token_id] *(batch_max_length - len(new_item)))

        inputs = torch.tensor(padded[:-1])     
        targets = torch.tensor(padded[1:]) 

        mask=targets==pad_token_id
        indices=torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]]=ignore_index

        if allowed_max_length is not None:
            inputs=inputs[:allowed_max_length]
            targets=targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor=torch.stack(inputs_lst).to(device)
    targets_tensor=torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

def cus_collate(device,allowed_max_length):
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    return customized_collate_fn