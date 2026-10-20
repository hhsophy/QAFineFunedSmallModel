
from torch.utils.data import Dataset
import tiktoken
tokenizer= tiktoken.get_encoding('gpt2')

def format_example(example):
    instruction = example["instruction"]
    input = example["question"]
    output = example["answer"]

    return (
        f"Context: {instruction}\n"
        f"Question: {input}\n"
        f"Answer: {output}"
    )


class InstructionDataset(Dataset):
    def __init__(self,data,tokenizer=tokenizer):
        self.data=data
        self.encoded_texts=[]

        for entry in data:
            instruction_plus_input=format_example(entry)
            self.encoded_texts.append(tokenizer.encode(instruction_plus_input))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)