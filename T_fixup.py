
import math
import torch
import torch.nn as nn


"""
T-Fixup initialization helper (implementation inspired by:
Huang et al., ICML 2020 - 'Improving Transformer Optimization Through Better Initialization').
References: paper + official repo. See: https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
and https://github.com/layer6ai-labs/T-Fixup.
"""


def apply_tfixup(model: nn.Module,
                 num_layers: int,
                 d_model: int,
                 encoder_only: bool = False,
                 zero_last_ff: bool = True,
                 verbose: bool = False):
    """
    Apply T-Fixup style initialization to a Transformer model.
    Args:
        model: nn.Module (transformer model instance)
        num_layers: N, number of encoder (or decoder) layers (the paper uses N for each stack)
        d_model: embedding dimension
        encoder_only: if True, apply encoder-scale formula to all (useful if you only have encoder)
        zero_last_ff: whether to zero the last linear of FFN blocks (Fixup-style)
        verbose: print summary lines for debugging
    Notes / Heuristics:
      - Xavier init for all linear projection matrices (nn.Linear)
      - Embedding weights: Gaussian (normal) with std = 1/sqrt(d_model) then scaled
      - Encoder scale factor ~ 0.67 * N^{-1/4}
      - Decoder scale factor ~ (9*N)^{-1/4}
      - Zero out the last projection in feed-forward blocks to keep residual small
      - Some models need small tweaks depending on naming of submodules; adapt as needed
    """

    # depth-dependent scaling factors (from Huang et al. slides / paper)
    encoder_scale = 0.67 * (num_layers ** (-1.0 / 4.0))
    decoder_scale = (9.0 * num_layers) ** (-1.0 / 4.0)

    if verbose:
        print(f"T-Fixup: encoder_scale={encoder_scale:.6f}, decoder_scale={decoder_scale:.6f}, d_model={d_model}")

    # helper to detect embedding modules
    def is_embedding(mod):
        return isinstance(mod, (nn.Embedding,))

    # iterate and initialize
    for name, mod in model.named_modules():
        # Embedding layers: Gaussian + scale
        if is_embedding(mod):
            # normal with std = 1/sqrt(d_model)
            std = 1.0 / math.sqrt(d_model)
            with torch.no_grad():
                if hasattr(mod, "weight"):
                    mod.weight.data.normal_(mean=0.0, std=std)
                    # choose scale: if it's decoder embeddings (name contains 'decoder') use decoder_scale
                    if not encoder_only and "decoder" in name.lower():
                        mod.weight.data.mul_(decoder_scale)
                        if verbose:
                            print(f"Scaled decoder embedding {name} by {decoder_scale:.6f}")
                    else:
                        mod.weight.data.mul_(encoder_scale)
                        if verbose:
                            print(f"Scaled encoder embedding {name} by {encoder_scale:.6f}")

            # if embedding has padding_idx, keep that row zeroed (optional)
            if getattr(mod, "padding_idx", None) is not None:
                with torch.no_grad():
                    mod.weight.data[mod.padding_idx].zero_()

        # Linear / projection matrices: Xavier init
        elif isinstance(mod, nn.Linear):
            with torch.no_grad():
                # Xavier uniform (common for projection matrices)
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0.0)

        # LayerNorm - if present, they usually set weight=1, bias=0
        elif isinstance(mod, nn.LayerNorm):
            with torch.no_grad():
                if hasattr(mod, "weight") and mod.weight is not None:
                    nn.init.constant_(mod.weight, 1.0)
                if hasattr(mod, "bias") and mod.bias is not None:
                    nn.init.constant_(mod.bias, 0.0)

    # Additional T-Fixup steps: scale some modules in encoder/decoder (heuristic approach)
    # We'll scale all parameters that belong to decoder stack names if present; otherwise scale encoder stack.
    # This is conservative: apply multipliers only to parameters with 'encoder'/'decoder' in their names.

    for name, param in model.named_parameters():
        if param.dim() == 0:
            continue
        lname = name.lower()
        if "decoder" in lname:
            # scale decoder params
            with torch.no_grad():
                param.data.mul_(decoder_scale)
        elif "encoder" in lname or encoder_only:
            with torch.no_grad():
                param.data.mul_(encoder_scale)
        # else: leave params that are neither encoder nor decoder (e.g., shared embeddings) as initialized above

    # Zeroing last projection in FFN blocks: heuristic
    # We try to detect typical names: 'fc2', 'linear2', 'ffn_output', 'feed_forward_output' etc.
    if zero_last_ff:
        zeroed = 0
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                lname = name.lower()
                # common patterns for the last linear of feed-forward: endswith 'fc2' or contains 'linear2' or 'ffn' and 'output'
                if lname.endswith(".fc2") or "linear2" in lname or ("ffn" in lname and "out" in lname) or lname.endswith(".ffn_output"):
                    with torch.no_grad():
                        mod.weight.zero_()
                        if mod.bias is not None:
                            mod.bias.zero_()
                        zeroed += 1
                        if verbose:
                            print(f"Zeroed FFN last linear: {name}")
        if verbose:
            print(f"T-Fixup: zeroed {zeroed} FFN last-linear layers (heuristic)")

    # done
    if verbose:
        print("T-Fixup initialization applied.")