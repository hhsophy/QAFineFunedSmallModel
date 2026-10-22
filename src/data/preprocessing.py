
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
    def __init__(self,data,context_size=1024,tokenizer=None):
        self.data=data
        self.encoded_texts=[]
        if tokenizer is None:
            self.tokenizer=tiktoken.get_encoding('gpt2')
        else:
            self.tokenizer=tokenizer

        for entry in data:
            context=f"Context: {entry["instruction"]}\n"
            question=f"Question: {entry["question"]} \n"
            answer=f"Answer: {entry["answer"]}"

            context_encoded=self.tokenizer.encode(context)
            question_encoded=self.tokenizer.encode(question)
            answer_encoded=self.tokenizer.encode(answer)
            context_legth=len(context_encoded)
            question_length=len(question_encoded)
            answer_length=len(answer_encoded)
        
            if context_legth + question_length + answer_length > context_size:
                if context_legth>question_length:
                    context_encoded=context_encoded[:context_size-(question_length+answer_length)]
                    context_legth=len(context_encoded)
                else:
                    question_encoded=question_encoded[:context_size-(context_legth+answer_length)]
                    question_length=len(question_encoded)
            encoded_text=context_encoded+question_encoded+answer_encoded
            prompt_length=context_legth+question_length
            example={"encoded_text":encoded_text,"prompt_length":prompt_length}
            #instruction_plus_input=format_example(entry)
            self.encoded_texts.append(example)
        
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)


def Data_Split(path,train_ratio=0.8,test_ratio=0.1):
        data=Load_data(path)
        Data=ConvToDict(data)[:1000]
        train_portion=int(len(Data)*train_ratio)
        test_portion=int(len(Data)*test_ratio)
        train_data=Data[:train_portion]
        test_data=Data[train_portion:train_portion+test_portion]
        val_data=Data[train_portion+test_portion:]

        return train_data,test_data,val_data