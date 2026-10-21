
from torch.utils.data import Dataset
import tiktoken
from .dataset import Load_data,ConvToDict

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
    def __init__(self,data,tokenizer=None):
        self.data=data
        self.encoded_texts=[]
        if tokenizer is None:
            self.tokenizer=tiktoken.get_encoding('gpt2')
        else:
            self.tokenizer=tokenizer

        for entry in data:
            instruction_plus_input=format_example(entry)
            self.encoded_texts.append(tokenizer.encode(instruction_plus_input))

    def __getitem__(self, index):
        answer=self.data[index]["answer"]
        input_without_answer=self.tokenizer.encode(format_example(self.data[index]).removesuffix(answer))
        
        return {"encoded":self.encoded_texts[index],"prom_len":len(input_without_answer)}
        
    def __len__(self):
        return len(self.data)


def Data_Split(path,train_ratio=0.8,test_ratio=0.1):
        data=Load_data(path)
        Data=ConvToDict(data)[:100]
        train_portion=int(len(Data)*train_ratio)
        test_portion=int(len(Data)*test_ratio)
        train_data=Data[:train_portion]
        test_data=Data[train_portion:train_portion+test_portion]
        val_data=Data[train_portion+test_portion:]

        return train_data,test_data,val_data