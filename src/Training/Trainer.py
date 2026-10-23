import torch
from src.evaluation import calc_loss_batch,evaluate_model
from src.Training.checkpoint import Save_checkpoint
from src.Training.utils import generate_and_print_simple
import time

def Fit(
    train_loader, val_loader, model, optimizer,
    device, scheduler, Sittings, tokenizer,Resume):
    num_epochs ,start_epoch= Sittings['num_epochs'],    Sittings.get('start_epoch', 0)
    global_step ,path= Sittings.get('global_step', -1), Sittings['checkpoint_path']
    sampler,save_point,srt_batch = Sittings['sampler'], Sittings['save_point'],Sittings.get('batch_idx', 0)
    eval_freq,eval_iter=Sittings['eval_freq'],Sittings['eval_iter']

    total_training_time = Sittings.get('total_training_time', 0.0) if Resume else 0.0
    session_start_time = time.perf_counter()

    train_losses, val_losses = [], []
    tokens_seen = 0
    start_context = 'The capital of Iraq is?'
    CheckPoint=Save_checkpoint(path)
    try:
        for epoch in range(start_epoch, num_epochs):
            print(f'\nStarting epoch {epoch + 1}/{num_epochs}')
            model.train()

            # The sampler resumes from its saved position.
            for batch_idx, batch in enumerate(train_loader,start=srt_batch):
                optimizer.zero_grad()

                inputs, targets = batch
                loss = calc_loss_batch(inputs, targets, model, device)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),max_norm=1.0)

                global_step += 1
                scheduler.CheckGlobal(global_step)
                scheduler.step(optimizer)
                optimizer.step()

                if global_step % save_point == 0:
                    CheckPoint.set_model(model)
                    CheckPoint.set_lora_only(model)
                    CheckPoint.set_optimizer(optimizer)
                    CheckPoint.set_scheduler(scheduler)
                    CheckPoint.set_rng()
                    CheckPoint.set_sampler(sampler)
                    CheckPoint.set_batch_idx(batch_idx)
                    CheckPoint.set_epoch(epoch)
                    CheckPoint.set_epoch_completed(False)
                    CheckPoint.set_global_step(global_step)
                    CheckPoint.set_train_loss(loss.item())
                    session_elapsed = time.perf_counter() - session_start_time
                    session_start_time = time.perf_counter()
                    current_total_time = total_training_time + session_elapsed
                    total_training_time=current_total_time
                    CheckPoint.set_train_time(total_training_time)
                    CheckPoint.save_checkpoint()
                    print(f'Saved mid-epoch checkpoint at step {global_step}')
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader,
                        device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f'Ep {epoch+1} (Iter {global_step:06d}): '
                        f'Train loss {train_loss:.3f}, '
                        f'Val loss {val_loss:.3f}'
                    )

                    
            print(f'Finished epoch {epoch + 1}/{num_epochs}')
            generate_and_print_simple(model,tokenizer,device,start_context)
            srt_batch=0

            # Save a separate checkpoint marking the completed epoch.
            CheckPoint.set_model(model)
            CheckPoint.set_lora_only(model)
            CheckPoint.set_optimizer(optimizer)
            CheckPoint.set_scheduler(scheduler)
            CheckPoint.set_rng()
            CheckPoint.set_sampler(sampler)
            CheckPoint.set_batch_idx(batch_idx)
            CheckPoint.set_epoch(epoch)
            CheckPoint.set_epoch_completed(True)
            CheckPoint.set_global_step(global_step)
            CheckPoint.set_train_loss(3.0)
            session_elapsed = time.perf_counter() - session_start_time
            session_start_time = time.perf_counter()
            current_total_time = total_training_time + session_elapsed
            total_training_time=current_total_time
            CheckPoint.set_train_time(total_training_time)
            CheckPoint.save_checkpoint()

            print(f'Saved ew epoch checkpoint at step {global_step}')
        return train_losses, val_losses
    
    except KeyboardInterrupt:
        print('\nTraining interrupted. Returning the latest losses...')
        return train_losses, val_losses