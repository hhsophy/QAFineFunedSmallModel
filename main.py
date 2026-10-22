import torch
import json
import tiktoken
from pathlib import Path
from src.data.collate import Create_Data_Loader
from src.data.preprocessing import Data_Split
from src.models.LR_scheduler import lr_scheduler_step
from src.models.model_loader import load_gpt,load_weights_into_gpt
from src.models.gpt_model import LLMModel
from src.models.LoRA import replace_linear_with_lora,FreezeModel
from src.Training.checkpoint import Load_Checkpoint
from src.Training.Trainer import Fit
from src.Training.utils import save_losses



def main(resume=False,CHOOSE_MODEL = "gpt2-small (124M)"):
    with open("D:/gpt2-qa/configs/qa.json", "r") as file:
        config = json.load(file)
    Sittings=config["Sittings"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(Sittings["seed"])



    Train_Data,Test_Data,Val_Data=Data_Split(config["data_path"],Sittings["train_portion"],Sittings["test_portion"])
    
    tokenizer=tiktoken.get_encoding("gpt2")    
    
    
    Train_loader,Train_sampler=Create_Data_Loader(Sittings,Train_Data,context_size=config["BASE_CONFIG"]["context_length"])
    Val_loader,Val_sampler=Create_Data_Loader(Sittings,Val_Data,context_size=config["BASE_CONFIG"]["context_length"])
    Test_loader,Test_sampler=Create_Data_Loader(Sittings,Test_Data,context_size=config["BASE_CONFIG"]["context_length"])
    
    Sittings["sampler"]=Train_sampler


    scheduler=lr_scheduler_step(Sittings,len(Train_loader))

    #params,BASE_CONFIG=load_gpt(config,CHOOSE_MODEL)
    BASE_CONFIG=config["BASE_CONFIG"]
    BASE_CONFIG.update(config["model_configs"][CHOOSE_MODEL])

    model=LLMModel(BASE_CONFIG)
    state_dict=torch.load(f"{config["gpt2place"]}/{CHOOSE_MODEL}.pt")
    model.load_state_dict(state_dict)
    #load_weights_into_gpt(model,params)
    
    FreezeModel(model)
    replace_linear_with_lora(model,rank=Sittings["rank"],alpha=Sittings["alpha"])
    model.to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=Sittings["initial_lr"], weight_decay=Sittings["weight_decay"])
    checkpoint_path = Path(Sittings["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    CheckPoint=Load_Checkpoint(checkpoint_path)
    if resume and checkpoint_path.exists():
            #CheckPoint.load_model(model)
            CheckPoint.load_lora_only(model)
            CheckPoint.load_optimizer(optimizer)
            CheckPoint.load_rng()
            CheckPoint.load_sampler(Train_sampler)
            Train_loader,Train_sampler=Create_Data_Loader(Sittings,Train_Data,context_size=config["BASE_CONFIG"]["context_length"])
            CheckPoint.load_scheduler(scheduler)
            sumdata=CheckPoint.return_rest()
            Sittings["sampler"]=Train_sampler
            Sittings["start_epoch"]=sumdata["start_epoch"]
            Sittings["global_step"]=sumdata["global_step"]
            Sittings["batch_idx"]=sumdata["batch_idx"]
            Sittings["total_training_time"]=sumdata["total_training_time"]
                                    
            print(f"Resuming from epoch {Sittings["start_epoch"]}")
            print(f"Global step: {Sittings["global_step"]}")
    else:
        print("Starting training from epoch 0")
    
    print (f"total steps : {len(Train_loader)*Sittings["num_epochs"]}")
    train_losses,val_losses,_=Fit(Train_loader,Val_loader,model,optimizer,device,scheduler,Sittings,tokenizer,resume)
    save_losses(train_losses=train_losses,val_losses=val_losses,path=config["results_path"],resume=resume)
    #return train_losses,val_losses,track_tokens_seen



if __name__ == "__main__":

    main(resume=False)