import os
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb
import config
from models.utils import get_model
from optim.base import train_base
import distributed
from datetime import datetime


def save_args_to_json(args: argparse.Namespace, filename: str) -> None:
    """
    Saves the arguments from an argparse Namespace to a JSON file, handling various non-serializable
    objects like torch.device, torch.Tensor, and numpy arrays.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        filename (str): The path where the JSON file will be saved.

    Returns:
        None
    """
    # Convert the argparse Namespace to a dictionary (deep copy to avoid modifying original)
    args_dict = vars(copy.deepcopy(args))
    
    # Handle non-serializable objects
    for key, value in args_dict.items():
        if isinstance(value, torch.device):
            args_dict[key] = str(value)  # Convert torch.device to string
        elif isinstance(value, torch.Tensor):
            args_dict[key] = value.tolist()  # Convert torch.Tensor to list
        elif isinstance(value, np.ndarray):
            args_dict[key] = value.tolist()  # Convert numpy arrays to list
        elif isinstance(value, set):
            args_dict[key] = list(value)  # Convert set to list
        elif hasattr(value, '__dict__'):  # Handle custom objects by converting them to dict if possible
            args_dict[key] = value.__dict__
        else:
            # Fallback: Try to serialize the value, otherwise convert it to a string
            try:
                json.dumps(value)
            except TypeError:
                args_dict[key] = str(value)

    # Save the dictionary to a JSON file
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)

def get_args() -> argparse.Namespace:
    """
    Parses command-line arguments using argparse. Handles configuration formats and additional arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Create an argument parser with the allow_abbrev flag disabled
    parser = argparse.ArgumentParser(allow_abbrev=False)
    
    # Add argument for selecting the config format
    parser.add_argument('--config_format', default='markov', choices=config.registered_formats())

    # Parse known arguments (those that match the defined ones)
    args, rem_args = parser.parse_known_args()

    # Use the parsed config format to process the remaining arguments with a base parser
    return config.parse_args_with_format(
        format=args.config_format, 
        base_parser=parser, 
        args=rem_args, 
        namespace=args
    )


def get_exp_name(args: argparse.Namespace) -> str:
    """
    Generates a descriptive name for the experiment based on various hyperparameters and settings.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The generated experiment name.
    """
    # Construct the experiment name string by concatenating hyperparameters and settings
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    
    exp_name = (
        f"{args.model}_lr{args.lr}_"
        f"rp{str(args.use_relative_positional_encoding)}_"
        f"bs{args.batch_size_per_chain}_nc{args.num_chains}_"
        f"ch{args.chain}_ord{args.order}_nl{args.n_layer}_nh{args.n_head}_"
        f"nhf{args.n_head_first_layer}_"
        f"const{args.construction}_"
        f"ne{args.n_embd}_sl{args.sequence_length}_sd{args.seed}_"
        f"opt{args.opt}_wu{args.warmup_percent}_wd{args.weight_decay}_dp{args.dropout}_"
        f"sch{args.scheduler}_b1{args.beta1}_b2{args.beta2}_gc{args.grad_clip}_"
        f"d{current_time}_bi{args.bias}_ef{args.eval_freq}"
    )

    # Append the wandb run prefix if specified
    if args.wandb_run_prefix != 'none':
        exp_name = args.wandb_run_prefix + '_' + exp_name

    # Add seed information
    exp_name += f"_seed={args.seed}"

    return exp_name


def main(args: argparse.Namespace) -> None:
    """
    Main function for training a Markov-based model with distributed learning support.
    
    Args:
        args (Namespace): Parsed command-line arguments.

    Returns:
        None
    """

    # Set the order and initialize the random generator
    order = args.order
    generator = torch.Generator(device=args.device)
    generator.seed()

    # Define the Markov transition probabilities (None for "icl" chain type)
    if args.chain == "icl":
        P = None
    else:
        raise NotImplementedError(f"Unknown chain type: {args.chain}.")

    # Enable TensorFloat32 (TF32) for faster training on supported hardware
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set default tensor dtype
    torch.set_default_dtype(args.dtype)

    # Set up distributed backend and adjust args for the specific process
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    # Set device type and handle CUDA device if applicable
    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    print(f"Loading dataset '{args.dataset}'")

    # Initialize the model and move it to the correct device
    model = get_model(args).to(args.device)  # Handle pretrained models if applicable
    model = distributed_backend.transform_model(model)

    # Set up parameter group specifications for optimization
    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    
    # Create a dictionary mapping parameter names to the actual model parameters
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    
    optimized_params_count = 0
    
    # Iterate through each parameter group
    for group in group_specs:
        params = []
        
        # Translate parameter names for distributed nodes
        for param_name in group["params"]:
            translated_param_names = distributed_backend.translate_model_parameter_name_for_node(param_name)
            
            # Append corresponding parameters using translated names
            params += [param_name_mapping[p_name] for p_name in translated_param_names]
        
        # Update the group with the correct parameter objects
        group["params"] = params
        
        # Sum the total number of elements (parameters) in the current group
        optimized_params_count += sum(p.numel() for p in group["params"])

    print(f"Number of optimized parameters: {optimized_params_count / 1e6:.2f}M")

    # Choose optimizer based on the command-line arguments
    if args.opt == 'adamw':
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"Using fused AdamW: {use_fused}")
        opt = torch.optim.AdamW(group_specs, lr=args.lr, betas=(args.beta1, args.beta2), 
                                weight_decay=args.weight_decay, fused=use_fused)
    else:
        opt = torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Set up learning rate scheduler if specified
    if args.scheduler != 'none':
        if args.scheduler in ['cos', 'linear']:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=args.lr, total_steps=args.iterations, 
                                                            pct_start=args.warmup_percent, anneal_strategy=args.scheduler, 
                                                            cycle_momentum=False, div_factor=100, final_div_factor=0.05)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    # Get world size for distributed training and generate experiment name
    args.world_size = distributed_backend.get_world_size()
    exp_name = get_exp_name(args)

    # Initialize wandb logging if enabled
    if distributed_backend.is_master_process() and args.wandb:
        params_copy = copy.deepcopy(vars(args))
        del params_copy['device']  # Remove non-serializable fields
        wandb.init(project=args.wandb_project, name=exp_name, config=params_copy)
        
    # Define checkpoint path
    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)
    elif os.path.isfile(os.path.join(ckpt_path, "summary.json")):  # If experiment is already completed
        print(f"Already found experiment '{ckpt_path}'.\nSkipping.")
        sys.exit(0)

    # Select the appropriate training function based on the model type
    if args.model == 'base':
        train = train_base
    else:
        raise NotImplementedError(f"No training method implemented for model type '{args.model}'.")

    print(f"\nTraining model={args.model} \n{vars(args)}\n")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Set model checkpoint path and save initial arguments
    model.ckpt_path = ckpt_path
    model.update_ckpt_path(model.ckpt_path)
    save_args_to_json(args, os.path.join(ckpt_path, "args.json"))

    # Train the model
    stats = train(model, opt, P, order, scheduler, args.iterations, args.acc_steps, 
                  args.batch_size_per_chain, args.num_chains, args.sequence_length, generator,
                  eval_freq=args.eval_freq, distributed_backend=distributed_backend,
                  ckpt_path=f"{ckpt_path}/ckpt.pt", extra_args=args)

    # Finalize and save stats
    args.device = None
    args.dtype = None
    stats['args'] = vars(args)
    if distributed_backend.is_master_process():
        with open(f"{ckpt_path}/summary.json", "w") as fs:
            json.dump(stats, fs)

    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    main(args)