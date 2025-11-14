from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import wandb
import time
import copy
from .plot_data import plot_markov_chain

from .utils import (
    optimal_est,
    eval,
    eval_probs,
    eval_att,
    get_batch,
    save_checkpoint,
)
from .utils import (
    get_true_transition_states,
    estimate_transition_states,
    compute_divergences,
)

from .data import get_random_P

import numpy as np
import json
import os
import json


def save_metrics_to_json(metrics, filename="metrics.json"):
    # Function to recursively convert Tensors to lists in the dictionary
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()  # Convert tensor to list
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_list(i) for i in obj]
        else:
            return obj

    # Convert the metrics to a JSON-serializable format
    serializable_metrics = tensor_to_list(metrics)

    # Save the list of metrics to a JSON file
    with open(filename, "w") as f:
        json.dump(serializable_metrics, f, indent=4)


def train_base(
    model,
    opt,
    P,
    order,
    scheduler,
    iterations,
    acc_steps,
    batch_size_per_chain,
    num_chains,
    sequence_length,
    generator,
    eval_freq,
    ckpt_path,
    distributed_backend,
    extra_args,
):
    device_type = "cuda" if "cuda" in str(extra_args.device) else "cpu"
    type_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    )  # extra_args.dtype) #changed!
    itr, substep, best_val_loss, text_table = (
        0,
        0,
        float("inf"),
        None,
    )  # best_val_loss not used atm, early stopping not recommended but possible

    metrics_list = []
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    batch_size = batch_size_per_chain * num_chains

    if not extra_args.no_compile:
        print(f"Compiling model ...")
        model = torch.compile(model)  # requires pytorch 2.0+

    alpha = 0.5 * torch.ones((batch_size, 2**order, 2))
    dist = Dirichlet(alpha)
    dist = None

    if P is not None:
        P_test = P
        print("Markov transition matrix:")
        print(P)
    else:
        P_test = get_random_P(
            order, 1, 1, generator, None, extra_args.device, extra_args.dtype
        ).squeeze(0)
        print("Test Markov transition matrix:")
        print(P_test)

        x_sample, y_sample = get_batch(
            None,
            order,
            sequence_length,
            1,
            50,
            generator,
            dist,
            extra_args,
            return_P=False,
        )

    # Optimal test loss
    opt_loss = optimal_est(P_test, order, sequence_length, generator, dist, extra_args)
    if extra_args.wandb:
        wandb.log(
            {
                "val/opt_loss": opt_loss,
            }
        )
    
    P_true_dict = get_true_transition_states(P_test, order=order)
    true_markov_path = f"{model.ckpt_path}/true_markov_chain/"
    os.makedirs(true_markov_path, exist_ok=True)
    wandb_path = "val"
    name = "true_markov_chain"
    plot_markov_chain(P_true_dict, name=name, plot_path= true_markov_path, wandb_path=wandb_path, enable_wandb=extra_args.wandb)

    model.train()

    t0 = time.time()
    
    num_sequences_to_log = 1 if extra_args.log_less else 5

    while itr < iterations:
        model.train()

        for microstep_idx in range(acc_steps):  # gradient accumulation
            x, y = get_batch(
                None,
                order,
                sequence_length,
                batch_size_per_chain,
                num_chains,
                generator,
                dist,
                extra_args,
                return_P=False,
            )
            with type_ctx:
                outputs = model(
                    x,
                    targets=y,
                )
            loss = outputs["loss"] / acc_steps
            loss.backward()
            substep += 1

        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1
        print(f"Training iteration {itr} | Loss: {loss.item()}")
        
        
        if (
            itr % eval_freq == 0 or itr == iterations or itr == 1
        ):  # from here it's only evaluation code, all the training is above
            # set_eval_mode(model)
            eval_time_start = time.time()
            model.eval()

            # Estimating the transition states
            # x, y = get_batch(
            #     None,
            #     order,
            #     sequence_length,
            #     batch_size_per_chain,
            #     num_chains,
            #     generator,
            #     dist,
            #     extra_args,
            #     return_P=False
            # )

            _ = model(x_sample, targets=y_sample, get_logits=True, attn_map_logging=True, num_sequences=num_sequences_to_log)

            t1 = time.time()
            dt = t1 - t0

            # Conventional Metrics

            train_loss = loss.detach().cpu().item()
            current_lr = (
                scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
            )
            val_acc, val_loss, val_perplexity = eval(
                model,
                P_test,
                order,
                sequence_length,
                batch_size_per_chain,
                num_chains,
                generator,
                extra_args,
                max_num_batches=10,
                ctx=type_ctx,
            )

            print_string = f"{itr} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
            print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
            if scheduler is not None:
                print_string += f" [lr] {current_lr:.5f}"
            print(print_string)

            metrics = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
                "val/opt_loss": opt_loss,
                "val/diff_opt_loss": abs(val_loss - opt_loss),
                "lr": current_lr,
                "val/iter": itr,
            }
            metrics_list.append(metrics)
            
            if extra_args.wandb:
                wandb.log(metrics)

            if itr == iterations or itr % eval_freq == 0:
                prob_vec, est_vec = eval_probs(
                    model,
                    P_test,
                    order,
                    sequence_length,
                    generator,
                    extra_args,
                    ctx=type_ctx,
                )
                
                name = "final_est" if itr == iterations else f"est"
                
                if extra_args.wandb:
                    for k in range(2**order):
                        for i in range(len(prob_vec[k])):
                            wandb.log(
                                {
                                    f"{name}/model_iter_{itr}_est_"
                                    + str(k): prob_vec[k][i].detach().cpu().item(),
                                    f"{name}/empirical_{itr}_est_"
                                    + str(k): est_vec[k][i].detach().cpu().item(),
                                }
                            )

                att_mean, att_std = eval_att(
                    model,
                    P_test,
                    order,
                    sequence_length,
                    batch_size_per_chain,
                    num_chains,
                    generator,
                    extra_args,
                    device=extra_args.device,
                    ctx=type_ctx,
                )
                if extra_args.wandb:
                    wandb.log(
                        {
                            "val/att_mean": att_mean,
                            "val/att_std": att_std,
                        }
                    )

            # set_train_mode(model)

            
            model.train()
            t0 = time.time()
            eval_time_end = time.time()
            print("Time taken for evaluation: ", eval_time_end - eval_time_start)
            
        model.update_iter()


    save_metrics_to_json(metrics_list, filename=f"{model.ckpt_path}/metrics.json")

    model.eval()
    try:
        model.create_training_gif()
    except Exception as e:
        print(f"Failed to create training gif - {e}")
    
    for i in range(10):
        folder_name = f"post_training-{i}"
        wandb_run_dir = wandb.run.dir
        ckpt_path_per_folder = f"{model.ckpt_path}/{folder_name}"
        print(f"Saving data to {ckpt_path_per_folder}")
        os.makedirs(ckpt_path_per_folder, exist_ok=True)
        alpha = 0.5 * torch.ones((1, 2**order, 2))
        dist = Dirichlet(alpha)
        dist = None
        P_test_i = get_random_P(
            order, 1, 1, generator, dist, extra_args.device, extra_args.dtype
        )
        x, y = get_batch(
            P_test_i, order, sequence_length, 1, 1, generator, dist, extra_args
        )
        
        true_markov_path = ckpt_path_per_folder
        os.makedirs(true_markov_path, exist_ok=True)
        
        print("Markov transition matrix:")
        print(P_test_i)
        true_P_dict = get_true_transition_states(P_test_i[0], order=order)
        wandb_path = folder_name
        name = "true_markov_chain"
        plot_markov_chain(true_P_dict, name=name, plot_path= true_markov_path, wandb_path=wandb_path, enable_wandb=extra_args.wandb)

        prob_vec, est_vec = eval_probs(
            model,
            P_test_i,
            order,
            sequence_length,
            generator,
            extra_args,
            ctx=type_ctx,
        )
        if extra_args.wandb:
            for k in range(2**order):
                for i in range(len(prob_vec[k])):
                    wandb.log(
                        {
                            f"{folder_name}/model_iter_{itr}_est_"
                            + str(k): prob_vec[k][i].detach().cpu().item(),
                            f"{folder_name}/empirical_{itr}_est_"
                            + str(k): est_vec[k][i].detach().cpu().item(),
                        }
                    )
        

        np.save(f"{ckpt_path_per_folder}/P_test.npy", P_test_i.cpu().numpy())
        # try:
        #     artifact_name_P_test = f"{folder_name}_P_test"
        #     artifact_P_test = wandb.Artifact(artifact_name_P_test, type="dataset")
        #     artifact_P_test.add_file(f'{ckpt_path_per_folder}/P_test.npy')
        #     wandb.log({artifact_name_P_test: artifact_P_test})
        # except Exception as e:
        #     print(f"Failed to log P_test image - {folder_name}, Error: {e}")

        # Save x.npy
        np.save(f"{ckpt_path_per_folder}/x.npy", x.cpu().numpy())
        # try:
        #     artifact_name_x = f"{folder_name}_x"
        #     artifact_x = wandb.Artifact(artifact_name_x, type="dataset")
        #     artifact_x.add_file(f'{ckpt_path_per_folder}/x.npy')
        #     wandb.log({artifact_name_x: artifact_x})
        # except Exception as e:
        #     print(f"Failed to log x image - {folder_name}, Error: {e}")

        np.save(f"{ckpt_path_per_folder}/y.npy", y.cpu().numpy())
        outputs = model(x, targets=y, folder_name=folder_name, save_forward=True)
        # try:
        #     artifact_name_est_vec = f"{folder_name}_est_vec"
        #     artifact_est_vec = wandb.Artifact(artifact_name_est_vec, type="dataset")
        #     artifact_est_vec.add_file(f'{ckpt_path_per_folder}/est_vec.pth')
        #     wandb.log({artifact_name_est_vec: artifact_est_vec})
        # except Exception as e:
        #     print(f"Failed to log est_vec image - {folder_name}, Error: {e}")

    save_path = f"{model.ckpt_path}/model.pth"

    print(f"saving checkpoint to {save_path}")

    if distributed_backend.is_master_process():
        print(f"saving checkpoint to {ckpt_path}")
        save_checkpoint(
            distributed_backend=distributed_backend,
            model=model,
            opt=opt,
            scheduler=scheduler,
            itr=itr,
            ckpt_path=save_path,
        )

    return stats
