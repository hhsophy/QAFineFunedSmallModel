import json
from pathlib import Path 
def Load_data(filepath):
    if Path.exists(filepath):
        with open(filepath,"r") as file:
            data=json.load(file)

        return data
    else:
        raise TypeError ("The data File not exists")


def ConvToDict(data):
    Data=[]
    if isinstance(data,list):
        raise TypeError("data file already in shape lsit[dict()]")
    else:
        row=data[list(data.keys())[0]]
        for i in range(len(row)):
            item={"instruction":data['context'][f"{i}"],"question":data['instruction'][f"{i}"],"answer":data['response'][f"{i}"]}
            Data.append(item)
        return Data