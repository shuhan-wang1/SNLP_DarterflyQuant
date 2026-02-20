import torch
import os
import argparse
import torch.nn as nn
import transformers
import functools
from tqdm import tqdm
import json
import sys
import gc
from accelerate import dispatch_model

sys.path.append('../fake_quant')

import model_utils
import data_utils
import rotation_utils
import utils

# 保存到数据盘而不是系统盘
default_path = '/root/autodl-tmp/data/train_data/'


def get_per_channel_scale(a, w, alpha):
    act_scales = torch.max(a.view(-1, a.shape[-1]).abs(), dim=0)[0].float().to('cuda')
    weight_scales = w.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5).float()
    scale = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)
    return scale.to('cpu')


def get_mutil_head_scale(a, w, alpha, config):
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    kv_head = config.num_key_value_heads
    act_scales = torch.max(a.view(-1, a.shape[-1]).abs(), dim=0)[0].float().to('cuda')
    weight_scales = w.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5).float()
    scale = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)
    scale = scale.reshape(
        kv_head,
        num_heads // kv_head,
        head_dim).max(dim=1)[0].squeeze()

    return scale.to('cpu')


def get_Rs_training_data(model, dataloader, save_path, args):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    config = model.config
    dtype = config.torch_dtype
    model_type = model_utils.model_type_extractor(model)
    dev = 'cuda'  # next(model.parameters()).device

    if model_type == model_utils.LLAMA_MODEL:
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        # for transformers >= 4.44.2,model.model has rotary emb
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
        layers = model.model.layers
    else:
        raise ValueError(
            "Only support for Llama now")

    layers[0] = layers[0].to(dev)
    inps = torch.zeros(
        (args.nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input   捕获第一层的输入
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if model_type == model_utils.LLAMA_MODEL:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise ValueError(
            "Only support for Llama now")

    # input of first layer for fp model
    fp_inps = inps

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    if attention_mask is not None:
        attention_mask = attention_mask.repeat(1, 1, 1, 1).to(dtype)
    else:
        attention_mask = None

    del inps, cache, dataloader, batch
    model.to('cpu')

    if args.smooth:
        smooth_scale = {}

    for idx, decoder_layer in tqdm(enumerate(layers), total=len(layers), desc="Processing layers"):
        decoder_layer = decoder_layer.to(dev)

        o_proj_inp = {'idx': 0}
        down_proj_inp = {'idx': 0}
        q_proj_inp = {'idx': 0}
        up_proj_inp = {'idx': 0}

        def stat_tensor(name, tensor):
            if 'o_proj' in name:
                o_proj_inp[o_proj_inp['idx']] = tensor.detach()
                o_proj_inp['idx'] += 1
            # elif 'down_proj' in name:
            #     down_proj_inp[down_proj_inp['idx']] = tensor.detach()
            #     down_proj_inp['idx'] += 1
            elif 'q_proj' in name:
                q_proj_inp[q_proj_inp['idx']] = tensor.squeeze().detach()
                q_proj_inp['idx'] += 1
            elif 'up_proj' in name:
                up_proj_inp[up_proj_inp['idx']] = tensor.squeeze().detach()
                up_proj_inp['idx'] += 1

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            stat_tensor(name, x)

        # 插入钩子函数
        hooks = []
        module_list = [
            'self_attn.q_proj',
            'self_attn.o_proj',
            'mlp.up_proj',
            'mlp.down_proj',
        ]
        for name, m in decoder_layer.named_modules():
            if isinstance(m, nn.Linear) and name in module_list:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name)))

        # 记录 layer idx 输出
        with torch.no_grad():
            for j in range(args.nsamples):
                fp_inps[j] = decoder_layer(fp_inps[j].unsqueeze(
                    0), attention_mask=attention_mask, position_ids=position_ids)[0]

        assert q_proj_inp['idx'] == args.nsamples, '检查一下代码，获取的样本数目不对。'
        assert up_proj_inp['idx'] == args.nsamples, '检查一下代码，获取的样本数目不对。'
        assert o_proj_inp['idx'] == args.nsamples, '检查一下代码，获取的样本数目不对。'
        # assert down_proj_inp['idx'] == args.nsamples, '检查一下代码，获取的样本数目不对。'

        del q_proj_inp['idx'], up_proj_inp['idx']
        for k, v in q_proj_inp.items():
            torch.save(q_proj_inp[k], os.path.join(
                save_path['r1'], f'sample_{k}_layer_{idx}_self_attn_q_proj.pt'))
            torch.save(up_proj_inp[k], os.path.join(
                save_path['r1'], f'sample_{k}_layer_{idx}_mlp_up_proj.pt'))

        o_proj_inp = [o_proj_inp[i] for i in range(args.nsamples)]  # [nsamples, 1, 2048, 4096]
        o_proj_inp = torch.cat(o_proj_inp, dim=0)  # [nsamples, 2048, 4096]
        # down_proj_inp = [down_proj_inp[i] for i in range(args.nsamples)]  # [nsamples, 1, 2048, 4096]
        # down_proj_inp = torch.cat(down_proj_inp, dim=0)  # [nsamples, 2048, 4096]

        torch.save(o_proj_inp, os.path.join(
            save_path['r2'], f'layer_{idx}_self_attn_o_proj.pt'))
        # torch.save(down_proj_inp, os.path.join(
        #     save_path['r4'], f'layer_{idx}_mlp_down_proj.pt'))

        if args.smooth:
            config = model.config

            smooth_scale[f'model.layers.{idx}.mlp.down_smooth'] = get_per_channel_scale(
                down_proj_inp, decoder_layer.mlp.down_proj.weight, args.alpha).cpu()
            smooth_scale[f'model.layers.{idx}.self_attn.o_smooth'] = get_mutil_head_scale(
                o_proj_inp, decoder_layer.self_attn.o_proj.weight, args.alpha, config).cpu()

        for h in hooks:
            h.remove()

        decoder_layer = decoder_layer.to('cpu')

    if args.smooth:
        torch.save(smooth_scale, os.path.join(save_path['smooth'], 'smooth_scale_0.85alpha.pt'))

    del fp_inps
    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = use_cache

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        choices=utils.supported_models,
                        help='model name')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--r_path', type=str, default='',
                        help='where to save the r1&2&4 training data')
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "ptb", "c4"],
                        help="Where to extract calibration data from.",)
    parser.add_argument('--alpha', type=float, default=0.85,
                        help='Smooth parameter.')
    parser.add_argument('--r_list', nargs='+', default=['r1', 'r2'],
                        help='Which Rs to get.')
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--smooth', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to get ov and mlp smooth scale.')
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    args = parser.parse_args()

    if args.r_path == '':
        r1_path = os.path.join(
            default_path, f"{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/r1_train")
        r2_path = os.path.join(
            default_path, f"{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/r2_train")
        r4_path = os.path.join(
            default_path, f"{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/r4_train")
        smooth_path = os.path.join(
            default_path, f"{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/smooth_scale")

    else:
        r1_path = os.path.join(
            args.r_path, f"{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/r1_train")
        r2_path = os.path.join(
            args.r_path, f"{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/r2_train")
        r4_path = os.path.join(
            args.r_path, f"{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/r4_train")
        smooth_path = os.path.join(
            args.r_path, f"{args.calib_dataset}_{args.nsamples}samples/{args.model.split('/')[-1]}/smooth_scale")
    
    args.r_path = {'r1': r1_path, 'r2': r2_path, 'r4': r4_path, }

    if not os.path.exists(args.r_path['r1']):
        os.makedirs(args.r_path['r1'])

    if not os.path.exists(args.r_path['r2']):
        os.makedirs(args.r_path['r2'])

    if not os.path.exists(args.r_path['r4']):
        os.makedirs(args.r_path['r4'])

    print(f"---> r1_path: {args.r_path['r1']}")
    print(f"---> r2_path: {args.r_path['r2']}")
    print(f"---> r4_path: {args.r_path['r4']}")

    if args.smooth:
        args.r_path['smooth'] = smooth_path
        print(f"---> smooth_path: {args.r_path['smooth']}")
        if not os.path.exists(args.r_path['smooth']):
            os.makedirs(args.r_path['smooth'])

    return args


@torch.no_grad()
def main():
    args = parse_args()
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)

    rotation_utils.fuse_layer_norms(model)
    dataloader = data_utils.get_loaders(
        args.calib_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=args.seqlen,
        eval_mode=False
    )
    get_Rs_training_data(model=model,
                         dataloader=dataloader,
                         save_path=args.r_path,
                         args=args)


if __name__ == '__main__':
    main()
