import torch
import torch.nn as nn

def calc_accuracy_loader(data_loader,model,device,num_batchs=None):
    if device.type !=(next(model.to(device).parameters()).device).type:
        model.to(device)

    model.eval()
    correct_predictions,num_examples=0,0
    if num_batchs is None:
        num_batchs=len(data_loader)
    else:
        num_batchs=min(num_batchs,len(data_loader))

    for i ,(input_batch,target_batch) in enumerate(data_loader):
        if i< num_batchs:
            input_batch=input_batch.to(device)
            target_batch=target_batch.to(device)

            with torch.no_grad():
                logits=model(input_batch)
            predicted_labels=torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels==target_batch).sum().item()
            )
        else:
            break
    return correct_predictions/num_examples


# calculate the loss values of the Batch
# this diff from the previuose one since this function calc the lossod the last token only
# for classification tasks
def calc_loss_batch(input_batch,target_batch,model,device):
    input_batch=input_batch.to(device)
    target_batch=target_batch.to(device)
    logits=model(input_batch)
    loss=nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

# same to the previouse one but it calc for number of batchs i.e. data_loader
def calc_loss_loader(data_loader,model, device, num_batchs=None):
    total_loss=0.
    if len(data_loader)==0:
        return float("nan")
    elif num_batchs is None:
        num_batchs=len(data_loader)
    else:
        num_batchs=min(num_batchs,len(data_loader))
    
    for i, (input_batch,target_batch) in enumerate(data_loader):
        if i < num_batchs:
            loss=calc_loss_batch(input_batch,target_batch,model,device)
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batchs



def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss=calc_loss_loader(train_loader,model,device,num_batchs=eval_iter)

        val_loss=calc_loss_loader(val_loader,model,device,num_batchs=eval_iter)

    model.train()
    return train_loss,val_loss


