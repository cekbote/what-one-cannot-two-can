"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import tiktoken
import torch
import wandb
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
import os
from PIL import Image
from moviepy import ImageSequenceClip


def modify_array_k(arr, k):
    arr_copy = arr.copy()  # Create a copy of the array
    rows, cols = arr.shape
    
    # Create a mask for k consecutive 1 values
    consecutive_k = np.ones((rows, cols - k + 1), dtype=bool)
    
    # Iterate to find k consecutive 1 values
    for i in range(k):
        consecutive_k &= arr[:, i:i + cols - k + 1] == 1
    
    # Shift the consecutive 1 values to the right by k indices
    right_shifted_k = np.zeros_like(arr, dtype=int)
    
    # Check if we can shift and adjust based on k
    if cols > k:
        right_shifted_k[:, k:] = consecutive_k[:, :cols - k]

    if k>=2:
        for shift in range(len(right_shifted_k)-(k-1)):
            right_shifted_k[(right_shifted_k[:, shift]==1), shift+1:shift+k] = 0
    
    # Set the first k 1 values to 0 where we found k consecutive 1s
    for i in range(k):
        arr_copy[:, i:i + cols - k + 1][consecutive_k] = 0
    
    # Set the k+1 value to 1 if k consecutive 1s were found
    arr_copy |= right_shifted_k
    
    return arr_copy

def normalize_data(data):
    try:
        # Ensure the data is a numpy array
        data = np.array(data)
        # Normalize between 0 and 1
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return normalized_data
    except Exception as e:
        return 0

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(ndim, device=device, dtype=dtype)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.construction = config.construction
        self.n_head = config.n_head_first_layer if (id == 0 and config.n_head_first_layer is not None) else config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias, **factory_kwargs)
        self.n_embd = config.n_embd
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, **factory_kwargs)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.id = id
        self.order = config.order
        self.iterations = config.iterations
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length, **factory_kwargs))
                                        .view(1, 1, config.sequence_length, config.sequence_length))
        self.memory = config.memory
        self.device = config.device
        self.wandb = config.wandb
        self.config = config
        self.iter = 1
        self.ckpt_path = None
        self.eval_freq = config.eval_freq
        self.images_path = {}
        self.sequence_length = config.sequence_length
        self.relative_positional_embedding = nn.Embedding(config.sequence_length, config.n_embd)
        self.relative_positions = torch.abs(torch.arange(config.sequence_length).unsqueeze(0) - torch.arange(config.sequence_length).unsqueeze(1)).to(config.device)
        self.use_relative_positional_encoding = config.use_relative_positional_encoding

        self.q_ln = nn.LayerNorm(config.n_embd)
        self.k_ln = nn.LayerNorm(config.n_embd)

    def get_qkv(self):
        q, k, v = self.c_attn.weight.T.split(self.n_embd, dim=1)
        return q, k, v


    def log_energy_and_weights(self, weight, wandb_name: str, iter: int) -> None:
        weight = weight.clone().detach()
        sv = torch.linalg.svdvals(weight)
        energy1 = sv[0]**2 / torch.sum(sv**2)
        energy2 = torch.sum(sv[:2]**2) / torch.sum(sv**2)
        energy3 = torch.sum(sv[:3]**2) / torch.sum(sv**2)
        energy4 = torch.sum(sv[:4]**2) / torch.sum(sv**2)
        energy5 = torch.sum(sv[:5]**2) / torch.sum(sv**2)
        if self.wandb:
            wandb.log({
                f"{wandb_name}-energy1": energy1.item(),
                f"{wandb_name}-energy2": energy2.item(),
                f"{wandb_name}-energy3": energy3.item(),
                f"{wandb_name}-energy4": energy4.item(),
                f"{wandb_name}-energy5": energy5.item(),
            })
        
        numpy_weight = weight.cpu().numpy()    
        self.save_weights(numpy_weight, wandb_name, iter)
    
    def visualize_weights(self, weight, wandb_name: str, iter: int) -> None:
        weight = weight.detach().clone()
        if self.wandb:
            wandb.log({f"{wandb_name}-iter{self.iter}-weight": wandb.Image(weight.cpu().numpy())})

    def save_weights(self, weight, wandb_name: str, iter: int, folder_name=False, save_folder=False, original_data=None) -> None:
        if save_folder and folder_name is not None:
            weight_folder_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/weights"
        else:
            weight_folder_path = f"{self.ckpt_path}/{wandb_name}/weights"
        os.makedirs(weight_folder_path, exist_ok=True)
        np.save(f"{weight_folder_path}/{wandb_name}-iter{self.iter}.npy", weight)
        if original_data is not None:
            np.save(f"{weight_folder_path}/{wandb_name}-iter{self.iter}-original.npy", original_data)
            np.savetxt(f"{weight_folder_path}/{wandb_name}-iter{self.iter}-original.txt", original_data)

    def compute_psudo_attention_map(self, original_data):
        k = self.order

        arrays = np.tril(np.tile(original_data, (len(original_data), 1)), 0)
        subsets = np.lib.stride_tricks.sliding_window_view(original_data, k)
        subsets_pad = arrays[0 : k-1, 0 : k]

        subsets = np.vstack((subsets_pad, subsets))

        # Ensure arrays and subsets have the same number of rows
        assert arrays.shape[0] == subsets.shape[0], "Each array row must have a corresponding subset row."

        # Get the lengths of the subsets (assuming all subsets have the same length)
        subset_len = subsets.shape[1]

        # Create sliding windows for all arrays (this results in a 3D array)
        sliding_windows = np.lib.stride_tricks.sliding_window_view(arrays, subset_len, axis=1)

        # Reshape subsets to make them compatible for broadcasting
        subsets = subsets[:, np.newaxis, :]  # This adds an extra dimension for broadcasting

        # Use broadcasting to compare each sliding window with the respective subset for each row
        matches = np.all(sliding_windows == subsets, axis=-1)  # Compare along the last axis
        
        matches = np.concatenate((np.zeros((np.shape(matches)[0], k)), matches * 1.0), axis=1)
        result = np.tril(matches[:, 0: len(original_data)], 0)

        return result / (np.sum(result, axis=1, keepdims=True) + 1e-9)
        
        
    def save_images(self, weight, wandb_name: str, iter: int, folder_name=False, save_folder=False, original_data=None) -> None:
        if save_folder and folder_name is not None:
            weight_images_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/images"
        else:
            weight_images_path = f"{self.ckpt_path}/{wandb_name}/images"
            
        os.makedirs(weight_images_path, exist_ok=True)

        # Create a subplot figure with 1 row and 3 columns (one for each heatmap)
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))  # Three subplots: weight, pseudo_attention_map, and abs_difference

        # Plot the weight heatmap
        heatmap = axes[0].imshow(weight, cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'Weight Heatmap ({self.iter})')
        if original_data is not None:
            axes[0].set_xticks(np.arange(original_data.shape[0]))
            axes[0].set_xticklabels(original_data, fontsize=5)
            axes[0].set_yticks(np.arange(original_data.shape[0]))
            axes[0].set_yticklabels(original_data, fontsize=5)
            axes[0].xaxis.set_ticks_position('top')

            # xticks = axes[0].get_xticks()
            # yticks = axes[0].get_yticks()
            # axes[0].vlines(x=xticks, ymin=0, ymax=original_data.shape[0] - 1, colors='gray', linestyles='--', alpha=0.5)
            # axes[0].hlines(y=yticks, xmin=0, xmax=original_data.shape[0] - 1, colors='gray', linestyles='--', alpha=0.5)
        fig.colorbar(heatmap, ax=axes[0])

        # Check if original_data exists and plot the pseudo attention map
        if original_data is not None:
            pseudo_attention_map = self.compute_psudo_attention_map(original_data)
        else:
            pseudo_attention_map = np.zeros_like(weight)  # Use a zero array if no original data

        # Plot the pseudo attention map
        heatmap2 = axes[1].imshow(pseudo_attention_map, cmap='viridis', interpolation='nearest')
        axes[1].set_title('Pseudo Attention Map')
        if original_data is not None:
            axes[1].set_xticks(np.arange(original_data.shape[0]))
            axes[1].set_xticklabels(original_data, fontsize=5)
            axes[1].set_yticks(np.arange(original_data.shape[0]))
            axes[1].set_yticklabels(original_data, fontsize=5)
            axes[1].xaxis.set_ticks_position('top')

            # xticks = axes[1].get_xticks()
            # yticks = axes[1].get_yticks()
            # axes[1].vlines(x=xticks, ymin=0, ymax=original_data.shape[0] - 1, colors='gray', linestyles='--', alpha=0.5)
            # axes[1].hlines(y=yticks, xmin=0, xmax=original_data.shape[0] - 1, colors='gray', linestyles='--', alpha=0.5)
        fig.colorbar(heatmap2, ax=axes[1])

        # Compute the absolute difference and plot it as the third heatmap
        abs_difference = np.abs(weight - pseudo_attention_map)
        heatmap3 = axes[2].imshow(abs_difference, cmap='viridis', interpolation='nearest')
        axes[2].set_title('Absolute Difference (Weight vs. Pseudo Attention)')
        if original_data is not None:
            axes[2].set_xticks(np.arange(original_data.shape[0]))
            axes[2].set_xticklabels(original_data, fontsize=5)
            axes[2].set_yticks(np.arange(original_data.shape[0]))
            axes[2].set_yticklabels(original_data, fontsize=5)
            axes[2].xaxis.set_ticks_position('top')
            
            # xticks = axes[2].get_xticks()
            # yticks = axes[2].get_yticks()
            # axes[2].vlines(x=xticks, ymin=0, ymax=original_data.shape[0] - 1, colors='gray', linestyles='--', alpha=0.5)
            # axes[2].hlines(y=yticks, xmin=0, xmax=original_data.shape[0] - 1, colors='gray', linestyles='--', alpha=0.5)
            
        fig.colorbar(heatmap3, ax=axes[2])

        # Save the combined figure
        combined_plot_path = f"{weight_images_path}/{wandb_name}-combined-iter{self.iter}.png"
        plt.savefig(combined_plot_path)
        plt.close()
        
        self.images_path[wandb_name].append(combined_plot_path) if wandb_name in self.images_path else self.images_path.update({wandb_name: [combined_plot_path]})

        # Log the combined plot to WandB
        if self.wandb:
            wandb.log({f"{wandb_name}/{wandb_name}-combined-iter{self.iter}": wandb.Image(combined_plot_path)})
            if save_folder and folder_name is not None:
                wandb.log({f"{folder_name}/{wandb_name}-combined-iter{self.iter}": wandb.Image(combined_plot_path)})
            else:
                wandb.log({f"{wandb_name}/{wandb_name}-combined-iter{self.iter}": wandb.Image(combined_plot_path)})

    def forward(self, x, get_att=True, folder_name=None, save_forward=False, original_data=None, attn_map_logging=False, num_sequences=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        if (self.iter == 1) or (self.iter % self.eval_freq == 0):
            q, k, v = self.get_qkv()
            self.log_energy_and_weights(q, f"q-id{self.id}", self.iter)
            self.log_energy_and_weights(k, f"k-id{self.id}", self.iter)
            self.log_energy_and_weights(v, f"v-id{self.id}", self.iter)
            
            proj_weight = self.c_proj.weight.T.detach().clone()
            self.log_energy_and_weights(proj_weight, f"proj-id{self.id}", self.iter)
            
            if self.c_proj.bias is not None:
                proj_bias = self.c_proj.bias.detach().clone()
                os.makedirs(f"{self.ckpt_path}/proj-id{self.id}", exist_ok=True)
                np.save(f"{self.ckpt_path}/proj-id{self.id}/proj-id{self.id}-bias-iter{self.iter}.npy", proj_bias.cpu().numpy())

        if self.iter == self.iterations + np.floor(self.iterations/self.config.eval_freq):
            print("id" + str(self.id) + " W_Q W_K W_V:")
            print(self.c_attn.weight.cpu().detach().type(torch.float).numpy())
            fig_qkv = self.c_attn.weight.cpu().detach().type(torch.float)
            fig_qkv = (fig_qkv - fig_qkv.min()) / (fig_qkv.max()-fig_qkv.min())
            plt.imshow(fig_qkv.numpy(), cmap='gray', interpolation='nearest')
            if self.wandb:
                wandb.log({"id" + str(self.id) + "-att-qkv-"+str(self.iter): plt})
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)

        C_per_head = q.size(-1) // self.n_head
        k = k.view(B, T, self.n_head, C_per_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C_per_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C_per_head).transpose(1, 2) # (B, nh, T, hs)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att_logged, rel_pos_addition, rel_pos_addition_v_head = None, 0, 0
        
        if self.flash and (not self.use_relative_positional_encoding):
            # efficient attention using Flash Attention CUDA kernels
            # Memory attention mask
            if self.memory >= 0:
                M1 = torch.ones(T, T, dtype=torch.bool).tril(diagonal=0)
                M2 = torch.ones(T, T, dtype=torch.bool).tril(diagonal=-self.memory-1)
                attn_mask = M1 * (~M2)
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(self.device), dropout_p=self.dropout)
            else:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            if self.use_relative_positional_encoding:
                _, k_rel_pos, rel_pos_addition_v = self.c_attn(self.relative_positional_embedding(self.relative_positions)).split(self.n_embd, dim=2)
                k_rel_pos_head = k_rel_pos.view(T, T, self.n_head, C_per_head).permute(2, 0, 1, 3)
                rel_pos_addition_v_head = rel_pos_addition_v.view(T, T, self.n_head, C_per_head).permute(2, 0, 1, 3)
                q_expanded = q.unsqueeze(-2).expand(-1, -1, -1, T, -1)
                rel_pos_addition = (q_expanded * k_rel_pos_head).sum(dim=-1)
            
            if self.id == 0:
                att = (q @ k.transpose(-2, -1) + rel_pos_addition) * (1.0 / math.sqrt(k.size(-1)))
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(self.device)  # Upper triangular matrix
            att = att.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                            
            att = F.softmax(att, dim=-1)
            
            att_logged = att.clone().detach().cpu().numpy()
            
            att = self.attn_dropout(att)
            
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        get_att = True if self.iter == 1 or self.iter % self.eval_freq == 0 or save_forward == True  or self.iter == self.iterations else False
        
        
        att_filtered = None
        
        if get_att:
            if att_logged  is None:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(torch.tril(torch.ones(T, T, device=self.device)).view(1, 1, T, T) == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att_filtered = att.clone().detach().cpu().numpy()
            else:
                att_filtered = att_logged
                
            att_mean = att_filtered.mean(axis=0)
            att_std = att_filtered.std(axis=0)

            num_sequences = num_sequences if num_sequences is not None else 1
            
            for seq in range(num_sequences):
                name_prefix = f"seq{seq}-" if attn_map_logging else ""
                
                if original_data is not None:
                    original_data_per_seq = original_data[seq, :].clone().detach().cpu().numpy()
                att_length = min(original_data_per_seq.shape[0], 64)
                
                for i in range(self.n_head):
                    per_head_id = str(self.id) + "-head" + str(i)
                    
                    if att_length > 64:
                        self.save_images(att_filtered[seq, i, 0:att_length, 0:att_length], "att-filtered-first-id" + per_head_id , self.iter, folder_name, save_forward, original_data[0:att_length])
                        self.save_images(att_filtered[seq, i, -att_length:, -att_length:], "att-filtered-last-id" + per_head_id , self.iter, folder_name, save_forward, original_data[-att_length:])
                    
                    if self.iter % (self.eval_freq*5) == 0 or self.iter == 1 or save_forward == True:
                        self.save_weights(att_filtered, name_prefix + "att-id" + per_head_id , self.iter, folder_name, save_forward, original_data_per_seq)
                    
                    if att_length <= 64:
                        self.save_images(att_filtered[seq, i, :, :], name_prefix + "att-id" + per_head_id , self.iter, folder_name, save_forward, original_data_per_seq)
                
                filter_max = lambda x, y: np.where(x > y, x - y, 0)
                
                if self.n_head == 2:
                    self.save_images(np.abs(att_filtered[seq, 0, :, :]-att_filtered[seq, 1, :, :]), "att-id" + str(self.id) + "diff", self.iter, folder_name, save_forward, original_data_per_seq)
                    self.save_images(filter_max(att_filtered[seq, 0, :, :], att_filtered[seq, 1, :, :]), "att-id" + str(self.id) + "max (0 minus 1)", self.iter, folder_name, save_forward, original_data_per_seq) 
                    self.save_images(filter_max(att_filtered[seq, 1, :, :], att_filtered[seq, 0, :, :]), "att-id" + str(self.id) + "max (1 minus 0)", self.iter, folder_name, save_forward, original_data_per_seq)
            
            if attn_map_logging:
                for i in range(self.n_head):
                    per_head_id = str(self.id) + "-head" + str(i)
                    att_filtered_mean = att_filtered[:, i, :, :].mean(axis=0)
                    att_filtered_std = att_filtered[:, i, :, :].std(axis=0)
                    self.save_images(
                        att_filtered_mean,
                        "avg_" + "att-id" + per_head_id,
                        self.iter,
                        folder_name,
                        save_forward,
                    )
                    
                    self.save_images(
                        att_filtered_std,
                        "std_" + "att-id" + per_head_id,
                        self.iter,
                        folder_name,
                        save_forward,
                    )

                if self.n_head == 2:
                    att_filtered_mean_0 = att_filtered[:, 0, :, :].mean(axis=0)
                    att_filtered_mean_1 = att_filtered[:, 1, :, :].mean(axis=0)
                    self.save_images(
                        np.abs(att_filtered_mean_0 - att_filtered_mean_1),
                        "avg_" + "att-id" + str(self.id) + "diff",
                        self.iter,
                        folder_name,
                        save_forward,
                    )
                    self.save_images(
                        filter_max(att_filtered_mean_0, att_filtered_mean_1),
                        "avg_" + "att-id" + str(self.id) + "max (0 minus 1)",
                        self.iter,
                        folder_name,
                        save_forward,
                    )
                    self.save_images(
                        filter_max(att_filtered_mean_1, att_filtered_mean_0),
                        "avg_" + "att-id" + str(self.id) + "min (1 minus 0)",
                        self.iter,
                        folder_name,
                        save_forward,
                    )
                    
        else:
            att_mean = None
            att_std = None

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if self.iter == self.iterations + np.floor(self.iterations/self.config.eval_freq):
            print("id" + str(self.id) + "_att_proj:")
            print(self.c_proj.weight)
        
        return y, att_filtered, att_std


    def create_training_gif(self):
        """
        Create GIFs for each key in the images_path dictionary with a 5-second delay per frame 
        using PIL and upload them to WandB.
        """
        for key, image_array in self.images_path.items():
            if key=="att-id1-head0" or key=="att-id0-head1" or key=="att-id0-head0":
                continue
            # Open images using PIL's Image module
            images = [Image.open(image_path) for image_path in image_array]

            # Define the path for saving the GIF
            gif_path = f"{self.ckpt_path}/{key}/{key}-iter{self.iter}.gif"

            # Save the GIF using PIL with 5 seconds per frame
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],  # Appending all images after the first one
                duration=3000,  # 5000 milliseconds = 5 seconds per frame
                loop=0,  # Loop forever
            )

            # Upload the GIF to wandb
            
            gif_name="training_gifs_seq" if "seq" in key else "training_gifs"
            video_name="videos_seq" if "seq" in key else "videos"
            
            wandb.log({f"{gif_name}/{key}": wandb.Image(gif_path)})

            mp4_path = f"{self.ckpt_path}/{key}/{key}-iter{self.iter}.mp4"
            clip = ImageSequenceClip([np.array(img) for img in images], fps=1/3)  # 3 seconds per frame = 0.33 fps
            clip.write_videofile(mp4_path, codec="libx264", fps=1/3)  # 3 seconds per frame = 0.33 fps
            wandb.log({f"{video_name}/{key}": wandb.Video(mp4_path)})
            

class MLP_new_construct(nn.Module):
    
    def __init__(self, id, config):
        super().__init__()
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        self.first_layer   = nn.Linear(config.n_embd, config.n_embd, bias=True, **factory_kwargs)
        self.second_layer  = nn.Linear(config.n_embd, config.n_embd, bias=False, **factory_kwargs)
        self.third_layer   = nn.Linear(config.n_embd, config.n_embd, bias=False, **factory_kwargs)
        self.fourth_layer  = nn.Linear(config.n_embd, config.n_embd, bias=False, **factory_kwargs)
        
        self.norm_first_layer = LayerNorm(config.n_embd, bias=True, **factory_kwargs)
        self.norm_second_layer = LayerNorm(config.n_embd, bias=False, **factory_kwargs)
        self.norm_third_layer = LayerNorm(config.n_embd, bias=False, **factory_kwargs)
        self.norm_fourth_layer = LayerNorm(config.n_embd, bias=False, **factory_kwargs)
        
        self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(config.dropout)
        self.id = id
        self.iterations = config.iterations
        self.config = config
        self.iter = 1
        self.eval_freq = config.eval_freq
        self.ckpt_path = None
        self.wandb = config.wandb
    
    def save_weights(self, weight, wandb_name: str, iter: int, folder_name, save_forward) -> None:
        if save_forward and folder_name is not None:
            weight_folder_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/weights"
        else:
            weight_folder_path = f"{self.ckpt_path}/{wandb_name}/weights"
        
        os.makedirs(weight_folder_path, exist_ok=True)
        np.save(f"{weight_folder_path}/{wandb_name}-iter{self.iter}.npy", weight)
    
    def forward(self, x, folder_name=None, save_forward=False):
       
        if self.iter == 1 or self.iter % self.eval_freq == 0 or save_forward == True: 
            weights_layer1 = self.first_layer.weight.clone().detach().cpu().numpy()
            weights_layer2 = self.second_layer.weight.clone().detach().cpu().numpy()
            weights_layer3 = self.third_layer.weight.clone().detach().cpu().numpy()
            weights_layer4 = self.fourth_layer.weight.clone().detach().cpu().numpy()
            self.save_weights(weights_layer1, "mlp-first_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(weights_layer2, "mlp-second_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(weights_layer3, "mlp-third_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(weights_layer4, "mlp-fourth_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
        
        x_first_layer =  self.norm_first_layer(self.activation(self.first_layer(x)))
        x = x + self.dropout(x_first_layer)
        
        x_second_layer = self.activation(self.second_layer(x))
        x = x_second_layer
        
        x_third_layer = self.norm_third_layer(self.activation(self.third_layer(x)))
        x = x + self.dropout(x_third_layer)
        
        x_fourth_layer = self.norm_fourth_layer(self.activation(self.fourth_layer(x)))
        x = x + self.dropout(x_fourth_layer)
        
        return x



class MLP_two_layers(nn.Module):
    
    def __init__(self, id, config):
        super().__init__()
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        self.first_layer   = nn.Linear(config.n_embd, config.n_embd, bias=True, **factory_kwargs)
        self.second_layer  = nn.Linear(config.n_embd, config.n_embd, bias=False, **factory_kwargs)
        self.third_layer   = nn.Linear(config.n_embd, config.n_embd, bias=False, **factory_kwargs)
        self.fourth_layer  = nn.Linear(config.n_embd, config.n_embd, bias=False, **factory_kwargs)
        
        self.norm_first_layer = LayerNorm(config.n_embd, bias=True, **factory_kwargs)
        self.norm_second_layer = LayerNorm(config.n_embd, bias=False, **factory_kwargs)
        self.norm_third_layer = LayerNorm(config.n_embd, bias=False, **factory_kwargs)
        self.norm_fourth_layer = LayerNorm(config.n_embd, bias=False, **factory_kwargs)
        
        self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(config.dropout)
        self.id = id
        self.iterations = config.iterations
        self.config = config
        self.iter = 1
        self.eval_freq = config.eval_freq
        self.ckpt_path = None
        self.wandb = config.wandb
    
    def save_weights(self, weight, wandb_name: str, iter: int, folder_name, save_forward) -> None:
        if save_forward and folder_name is not None:
            weight_folder_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/weights"
        else:
            weight_folder_path = f"{self.ckpt_path}/{wandb_name}/weights"
        
        os.makedirs(weight_folder_path, exist_ok=True)
        np.save(f"{weight_folder_path}/{wandb_name}-iter{self.iter}.npy", weight)
    
    def forward(self, x, folder_name=None, save_forward=False):
       
        if self.iter == 1 or self.iter % self.eval_freq == 0 or save_forward == True: 
            weights_layer1 = self.first_layer.weight.clone().detach().cpu().numpy()
            weights_layer2 = self.second_layer.weight.clone().detach().cpu().numpy()
            weights_layer3 = self.third_layer.weight.clone().detach().cpu().numpy()
            weights_layer4 = self.fourth_layer.weight.clone().detach().cpu().numpy()
            self.save_weights(weights_layer1, "mlp-first_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(weights_layer2, "mlp-second_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(weights_layer3, "mlp-third_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(weights_layer4, "mlp-fourth_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)

        
        x_first_layer =  self.norm_first_layer(self.activation(self.first_layer(x)))
        x = x + self.dropout(x_first_layer)
        
        x_second_layer = self.second_layer(x)
        x = x_second_layer
        
        return x

class MLP_three_layers(nn.Module):
    
    def __init__(self, id, config):
        super().__init__()
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        self.first_layer   = nn.Linear(config.n_embd, config.n_embd, bias=True, **factory_kwargs)
        self.second_layer  = nn.Linear(config.n_embd, config.n_embd, bias=False, **factory_kwargs)
        self.third_layer   = nn.Linear(config.n_embd, config.n_embd, bias=False, **factory_kwargs)
        self.fourth_layer  = nn.Linear(config.n_embd, config.n_embd, bias=False, **factory_kwargs)
        
        self.norm_first_layer = LayerNorm(config.n_embd, bias=True, **factory_kwargs)
        self.norm_second_layer = LayerNorm(config.n_embd, bias=False, **factory_kwargs)
        self.norm_third_layer = LayerNorm(config.n_embd, bias=False, **factory_kwargs)
        self.norm_fourth_layer = LayerNorm(config.n_embd, bias=False, **factory_kwargs)
        
        self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(config.dropout)
        self.id = id
        self.iterations = config.iterations
        self.config = config
        self.iter = 1
        self.eval_freq = config.eval_freq
        self.ckpt_path = None
        self.wandb = config.wandb
    
    def save_weights(self, weight, wandb_name: str, iter: int, folder_name, save_forward) -> None:
        if save_forward and folder_name is not None:
            weight_folder_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/weights"
        else:
            weight_folder_path = f"{self.ckpt_path}/{wandb_name}/weights"
        
        os.makedirs(weight_folder_path, exist_ok=True)
        np.save(f"{weight_folder_path}/{wandb_name}-iter{self.iter}.npy", weight)
    
    def forward(self, x, folder_name=None, save_forward=False):
       
        if self.iter == 1 or self.iter % self.eval_freq == 0 or save_forward == True: 
            weights_layer1 = self.first_layer.weight.clone().detach().cpu().numpy()
            weights_layer2 = self.second_layer.weight.clone().detach().cpu().numpy()
            weights_layer3 = self.third_layer.weight.clone().detach().cpu().numpy()
            weights_layer4 = self.fourth_layer.weight.clone().detach().cpu().numpy()
            self.save_weights(weights_layer1, "mlp-first_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(weights_layer2, "mlp-second_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(weights_layer3, "mlp-third_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(weights_layer4, "mlp-fourth_layer-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)

        
        x_original = x 
        x_1 =  x + self.norm_first_layer(self.activation(self.first_layer(x)))
        x_2 = x + self.norm_second_layer(self.second_layer(x_1)) 
        x_3  = x_2 + self.norm_third_layer(self.third_layer(x_2))
        
        return x_3
        
class MLP(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias, **factory_kwargs)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, **factory_kwargs)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        self.id = id
        self.iterations = config.iterations
        self.config = config
        self.iter = 1
        self.eval_freq = config.eval_freq
        self.ckpt_path = None
        self.wandb = config.wandb

    def save_weights(self, weight, wandb_name: str, iter: int, folder_name, save_forward) -> None:
        if save_forward and folder_name is not None:
            weight_folder_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/weights"
        else:
            weight_folder_path = f"{self.ckpt_path}/{wandb_name}/weights"
        
        os.makedirs(weight_folder_path, exist_ok=True)
        np.save(f"{weight_folder_path}/{wandb_name}-iter{self.iter}.npy", weight)

    def forward(self, x, folder_name=None, save_forward=False):

        weights_layer1 = self.c_fc.weight.clone().detach().cpu().numpy()
        weights_layer2 = self.c_proj.weight.clone().detach().cpu().numpy()
        if self.iter == 1 or self.iter % self.eval_freq == 0 or save_forward == True:
            self.save_weights(weights_layer1, "mlp-c_fc-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(weights_layer2, "mlp-c_proj-id" + str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        if self.iter == self.iterations + np.floor(self.iterations/self.config.eval_freq):
            print("id" + str(self.id) + "_c_fc:")
            print(self.c_fc.weight)
            print("id" + str(self.id) + "_c_proj:")
            print(self.c_proj.weight)

        return x


class Block(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias, **factory_kwargs)
        self.attn = CausalSelfAttention(id, config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias, **factory_kwargs)
        self.mlp = MLP(id, config)
        self.construction = config.construction
        self.mlp = MLP_new_construct(id, config)
        self.id = id

    def forward(self, x, get_att=False, save_forward=False, folder_name=None, original_data = None, attn_map_logging=False, num_sequences=None):
        z, att_mean, att_std = self.attn(x, get_att=get_att, folder_name=folder_name, save_forward=save_forward, original_data=original_data, attn_map_logging=attn_map_logging, num_sequences=num_sequences)
        x_tilde = x + z
        if self.id == 0:
            x = self.mlp(x_tilde, folder_name=folder_name, save_forward=save_forward)
        else:
            x = x_tilde                
        return x, att_mean, att_std
    

class GPTBase(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.wandb = config.wandb
        self.iterations = config.iterations
        self.iter = 1
        self.eval_freq = config.eval_freq
        self.ckpt_path = None
        self.data = None
        self.use_relative_positional_encoding = config.use_relative_positional_encoding
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, **factory_kwargs),
            wpe = nn.Embedding(config.sequence_length, config.n_embd, **factory_kwargs),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(id, config) for id in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias, **factory_kwargs),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True, **factory_kwargs)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        if self.config.init == "ashok":
            for pn, p in self.named_parameters():
                if pn.endswith('mlp.c_fc.weight'):
                    torch.nn.init.constant_(p, config.init_value)
                elif pn.endswith('mlp.c_proj.weight'):
                    torch.nn.init.constant_(p, -config.init_value)
                elif pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        else:
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def create_training_gif(self):
        for block in self.transformer.h:
            block.attn.create_training_gif()
            
    def update_ckpt_path(self, ckpt_path):
        for block in self.transformer.h:
            block.attn.ckpt_path = ckpt_path
            block.mlp.ckpt_path = ckpt_path
    
    def update_iter(self):
        self.iter += 1
        for block in self.transformer.h:
            block.attn.iter = self.iter
            block.mlp.iter = self.iter

    def save_weights(self, weight, wandb_name: str, iter: int, folder_name=None, save_forward=False) -> None:
        if save_forward and folder_name is not None:
            weight_folder_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/weights"
        else:
            weight_folder_path = f"{self.ckpt_path}/{wandb_name}/weights"
        os.makedirs(weight_folder_path, exist_ok=True)
        np.save(f"{weight_folder_path}/{wandb_name}-iter{self.iter}.npy", weight)

    def save_images(self, weight, wandb_name: str, iter: int, folder_name=None, save_forward=False) -> None:
        if save_forward and folder_name is not None:
            weight_images_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/images"
            try:
                wandb.log({f"{folder_name}/{wandb_name}-iter{self.iter}-weight": plt.imshow(weight, cmap='viridis', interpolation='nearest')})
            except Exception as e:
                print(f"Error: {e}")
        else:
            weight_images_path = f"{self.ckpt_path}/{wandb_name}/images"
        os.makedirs(weight_images_path, exist_ok=True)
        plt.figure()
        heatmap=plt.imshow(weight, cmap='viridis', interpolation='nearest')
        plt.colorbar(heatmap)
        plt.savefig(f"{weight_images_path}/{wandb_name}-iter{self.iter}.png")
        plt.close()
        
        if self.wandb:
            wandb.log({f"{wandb_name}/{wandb_name}-iter{self.iter}": wandb.Image(f"{weight_images_path}/{wandb_name}-iter{self.iter}.png")})

    def plot_images_on_wandb(self, weight, wandb_name: str, iter: int) -> None:
        if self.wandb:
            wandb.log({f"{wandb_name}-iter{self.iter}-weight": plt.imshow(weight, cmap='viridis', interpolation='nearest')}) 

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, get_logits=False, get_att=True, folder_name=None, save_forward = False, attn_map_logging=False, num_sequences=None):
        device = idx.device
        b, t = idx.size()
        if self.iter == self.iterations + np.floor(self.iterations/self.config.eval_freq):
            print("Input sequence (first 100 samples):")
            print(idx[0,:100])
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        if self.iter == 1 or self.iter % self.eval_freq == 0 or save_forward == True:
            copied_token_embedding_weights = self.transformer.wte.weight.clone().detach().cpu().numpy()
            # self.plot_images_on_wandb(copied_token_embedding_weights, "token_embeddings/tok_emb", self.iter)
            copied_token_embedding_weights = normalize_data(copied_token_embedding_weights)
            self.save_images(copied_token_embedding_weights, "token_embeddings", self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(copied_token_embedding_weights, "token_embeddings", self.iter, folder_name=folder_name, save_forward=save_forward)
                        
            copied_positional_embedding_weights = self.transformer.wpe.weight.clone().detach().cpu().numpy()
            if self.wandb:
                wandb.log({f"positional_embeddings/pos_emb-iter{self.iter}": plt.imshow(copied_positional_embedding_weights,
                                                                                    cmap='viridis', interpolation='nearest')})
            # self.plot_images_on_wandb(copied_positional_embedding_weights, "positional_embeddings/pos_emb", self.iter)
            copied_positional_embedding_weights = normalize_data(copied_positional_embedding_weights)
            self.save_images(copied_positional_embedding_weights, "positional_embeddings", self.iter)
            self.save_weights(copied_positional_embedding_weights, "positional_embeddings", self.iter)
        
        if self.iter == self.iterations + np.floor(self.iterations/self.config.eval_freq):
            print("wte:")
            print(self.transformer.wte.weight)
            print("wpe:")
            print(self.transformer.wpe.weight)

        if self.use_relative_positional_encoding:
            x = self.transformer.drop(tok_emb)
        else:
            x = self.transformer.drop(tok_emb + pos_emb)
        
        attentions = []
        filter_max = lambda x, y: np.where(x > y, x - y, 0)
        
        for block in self.transformer.h:
            x, att_filtered, att_std = block(x, get_att=get_att, folder_name=folder_name, save_forward=save_forward, original_data=idx, attn_map_logging=attn_map_logging, num_sequences=num_sequences)
            if attn_map_logging:
                attentions.append(att_filtered)
            
        
        if attn_map_logging:
            idx_converted = idx.clone().detach().cpu().numpy()
            for seq in range(num_sequences):
                name_prefix = f"bw_layers-l1h0-l0h0-seq{seq}_" if attn_map_logging else ""
                original_data_per_seq = idx_converted[seq, :]
                att_head_layer_0_head_0 = attentions[0][seq, 0, :, :]
                att_head_layer_1_head_0 = attentions[1][seq, 0, :, :]
                
                self.transformer.h[0].attn.save_images(
                    np.abs(att_head_layer_1_head_0 - att_head_layer_0_head_0),
                    name_prefix + "att_diff",
                    self.iter,
                    folder_name,
                    save_forward,
                    original_data_per_seq,
                )
                
                self.transformer.h[0].attn.save_images(
                    filter_max(att_head_layer_1_head_0, att_head_layer_0_head_0),
                    name_prefix + "att_max (l1h0 minus l0h0)",
                    self.iter,
                    folder_name,
                    save_forward,
                    original_data_per_seq,
                )
                
                self.transformer.h[0].attn.save_images(
                    filter_max(att_head_layer_0_head_0, att_head_layer_1_head_0),
                    name_prefix + "att_max (l0h0 minus l1h0)",
                    self.iter,
                    folder_name,
                    save_forward,
                    original_data_per_seq,
                )
        
            att_head_layer_0_head_0_mean = np.mean(attentions[0][:, 0, :, :], axis=0)
            att_head_layer_1_head_0_mean = np.mean(attentions[1][:, 0, :, :], axis=0)
            
            name_prefix = f"bw_layers-l1h0-l0h0_"
            
            self.transformer.h[0].attn.save_images(
                np.abs(att_head_layer_1_head_0_mean - att_head_layer_0_head_0_mean),
                name_prefix + "avg_" + "att-diff",
                self.iter,
                folder_name,
                save_forward,
            ) 
            
            self.transformer.h[0].attn.save_images(
                filter_max(att_head_layer_1_head_0_mean, att_head_layer_0_head_0_mean),
                name_prefix + "avg_" + "att-max (l1h0 minus l0h0)",
                self.iter,
                folder_name,
                save_forward,
            )
            
            self.transformer.h[0].attn.save_images(
                filter_max(att_head_layer_0_head_0_mean, att_head_layer_1_head_0_mean),
                name_prefix + "avg_" + "att-max (l0h0 minus l1h0)",
                self.iter,
                folder_name,
                save_forward,
            )
            
            if attentions[0].shape[1] == 2:

                for seq in range(num_sequences):
                    name_prefix = f"bw_layers-l1h0-l0h1-seq{seq}_" if attn_map_logging else ""
                    original_data_per_seq = idx_converted[seq, :]
                    att_head_layer_0_head_1 = attentions[0][seq, 1, :, :]
                    att_head_layer_1_head_0 = attentions[1][seq, 0, :, :]
                    
                    self.transformer.h[0].attn.save_images(
                        np.abs(att_head_layer_1_head_0 - att_head_layer_0_head_1),
                        name_prefix + "att_diff",
                        self.iter,
                        folder_name,
                        save_forward,
                        original_data_per_seq,
                    )
                    
                    self.transformer.h[0].attn.save_images(
                        filter_max(att_head_layer_1_head_0, att_head_layer_0_head_1),
                        name_prefix + "att_max (l1h0 minus l0h1)",
                        self.iter,
                        folder_name,
                        save_forward,
                        original_data_per_seq,
                    )
                    
                    self.transformer.h[0].attn.save_images(
                        filter_max(att_head_layer_0_head_1, att_head_layer_1_head_0),
                        name_prefix + "att_max (l0h1 minus l1h0)",
                        self.iter,
                        folder_name,
                        save_forward,
                        original_data_per_seq,
                    )
                
                
                name_prefix = f"bw_layers-l1h0-l0h1_"
                
                att_head_layer_0_head_1_mean = np.mean(attentions[0][:, 1, :, :], axis=0)
                
                self.transformer.h[0].attn.save_images(
                    np.abs(att_head_layer_1_head_0_mean - att_head_layer_0_head_1_mean),
                    name_prefix + "avg_" + "att-diff",
                    self.iter,
                    folder_name,
                    save_forward,
                )
                
                self.transformer.h[0].attn.save_images(
                    filter_max(att_head_layer_1_head_0_mean, att_head_layer_0_head_1_mean),
                    name_prefix + "avg_" + "att-max (l1h0 minus l0h1)",
                    self.iter,
                    folder_name,
                    save_forward,
                )
                
                self.transformer.h[0].attn.save_images(
                    filter_max(att_head_layer_0_head_1_mean, att_head_layer_1_head_0_mean),
                    name_prefix + "avg_" + "att-max (l0h1 minus l1h0)",
                    self.iter,
                    folder_name,
                    save_forward,
                )
                

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) # (b, t, vocab_size)
            if self.iter == self.iterations + np.floor(self.iterations/self.config.eval_freq):
                print("lm_head:")
                print(self.lm_head.weight)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        logits = F.softmax(logits, dim=-1)
        logits = logits if get_logits else None
        att_mean = None
        att_std = att_std if get_att else None

        return {'logits': logits, 'loss': loss, 'att_mean': att_mean, 'att_std': att_std}

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:sequence_length])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:sequence_length,:sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        pass

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')
            
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1,-1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
