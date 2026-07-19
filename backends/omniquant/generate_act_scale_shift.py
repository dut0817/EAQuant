import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
import argparse
import torch.nn as nn

from datasets import load_dataset
import functools
from tqdm import tqdm
from datautils import get_loaders
try:
    from llava.model import *   # required for llava
except ImportError:
    print("If want to quantize llave models, you should manually install llava from https://github.com/haotian-liu/LLaVA")

# import pdb



def get_act_scales(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_scales

def get_act_shifts(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))


    for h in hooks:
        h.remove()

    return act_shifts




def build_model_and_tokenizer(model_name):
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    config = AutoConfig.from_pretrained(model_name, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        legacy=config.model_type == "mistral",
        padding_side="right",
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model name or path')
    parser.add_argument('--net', type=str, default=None,
                        help='name used for output act scale/shift files; defaults to model path basename')
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/',
                        help='where to save the act scales')
    parser.add_argument('--shifts-output-path', type=str, default='./act_shifts/',
                        help='where to save the act shifts')
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix", "pile", "medmix"],
        help="Where to extract calibration data from.",)
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model)
    dataloader, _ = get_loaders(
    args.calib_dataset,
    nsamples=args.num_samples,
    seed=args.seed,
    model=args.model,
    seqlen=args.seq_len,
    )
    
    if args.net is None:
        args.net = args.model.split('/')[-1]
    act_scales = get_act_scales(model, dataloader,args.num_samples)
    save_path = os.path.join(args.scales_output_path,f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

    act_shifts = get_act_shifts(model, dataloader,args.num_samples)
    save_path = os.path.join(args.shifts_output_path,f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_shifts, save_path)


if __name__ == '__main__':
    main()
