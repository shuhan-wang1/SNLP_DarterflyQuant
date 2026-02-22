"""
Unified argument parser for DartQuant v2.

Key required arguments (no defaults):
  --loss {whip, swd_unif, swd_gauss}
  --quantizer_type {int4, nf4}

Optional new argument:
  --butterfly (applies learnable Butterfly Givens to R3/R4 only)
"""

import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DartQuant v2: Unified One-Click Quantization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ==== Model ====
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--model', type=str, required=True,
        help='HuggingFace model name (e.g., meta-llama/Llama-3.2-1B)')
    model_group.add_argument('--hf_token', type=str, default=None,
                             help='HuggingFace access token')
    model_group.add_argument(
        '--cache_dir', type=str, default='/root/autodl-tmp/huggingface',
        help='Cache directory for HuggingFace model weights. '
             'Default matches scripts/stat_and_download.py so models already '
             'downloaded by that script are reused without re-downloading.')
    model_group.add_argument(
        '--datasets_cache_dir', type=str, default='/root/autodl-tmp/datasets',
        help='Cache directory for HuggingFace datasets. '
             'Default matches scripts/stat_and_download.py.')
    model_group.add_argument('--seed', type=int, default=0,
                             help='Random seed')

    # ==== Loss Function (REQUIRED) ====
    loss_group = parser.add_argument_group('Loss Function')
    loss_group.add_argument(
        '--loss', type=str, required=True,
        choices=['whip', 'swd_unif', 'swd_gauss'],
        help='Loss function for rotation training. '
             'whip: original DartQuant exponential repulsion. '
             'swd_unif: Sliced Wasserstein Distance to Uniform (pairs with int4). '
             'swd_gauss: Gaussian SWD (pairs with nf4). '
             'REQUIRED - no default.')

    # ==== Quantizer Type (REQUIRED) ====
    quant_group = parser.add_argument_group('Quantizer Type')
    quant_group.add_argument(
        '--quantizer_type', type=str, required=True,
        choices=['int4', 'nf4'],
        help='Quantization method. '
             'int4: standard uniform INT4 quantization (GPTQ/RTN). '
             'nf4: Normal Float 4-bit via bitsandbytes (weight-only). '
             'REQUIRED - no default.')

    # ==== Rotation Options ====
    rot_group = parser.add_argument_group('Rotation Options')
    rot_group.add_argument(
        '--butterfly', action='store_true', default=False,
        help='Use learnable Butterfly Givens rotations for R3 and R4 '
             'instead of fixed random Hadamard. Only affects R3/R4, '
             'never R1/R2.')
    rot_group.add_argument(
        '--use_r1', action='store_true', default=True,
        help='Use R1 global rotation')
    rot_group.add_argument(
        '--no_r1', action='store_false', dest='use_r1',
        help='Disable R1')
    rot_group.add_argument(
        '--use_r2', type=str, default='offline',
        choices=['offline', 'online', 'none'],
        help='R2 rotation mode')
    rot_group.add_argument(
        '--use_r3', action='store_true', default=True,
        help='Use R3 for Q/K online rotation')
    rot_group.add_argument(
        '--no_r3', action='store_false', dest='use_r3',
        help='Disable R3')
    rot_group.add_argument(
        '--use_r4', action='store_true', default=True,
        help='Use R4 for down-projection online rotation')
    rot_group.add_argument(
        '--no_r4', action='store_false', dest='use_r4',
        help='Disable R4')
    rot_group.add_argument(
        '--r1_path', type=str, default=None,
        help='Path to pre-trained R1 rotation matrix (skip R1 training)')
    rot_group.add_argument(
        '--r2_path', type=str, default=None,
        help='Path to pre-trained R2 rotation matrices (skip R2 training)')
    rot_group.add_argument(
        '--rotate_mode', type=str, default='hadamard',
        choices=['hadamard', 'random'],
        help='Initialization mode for R1/R2 (if not loading from path)')
    rot_group.add_argument(
        '--smooth', type=str, default=None,
        help='Path to pre-computed SmoothQuant scales')

    # ==== Training Hyperparameters ====
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--lr', type=float, default=1e-3,
                             help='Learning rate for rotation training')
    train_group.add_argument('--momentum', type=float, default=0.9,
                             help='SGD momentum')
    train_group.add_argument('--r1_epochs', type=int, default=10,
                             help='R1 training epochs')
    train_group.add_argument('--r2_epochs', type=int, default=5,
                             help='R2 training epochs')
    train_group.add_argument('--butterfly_epochs', type=int, default=10,
                             help='Butterfly R3/R4 training epochs')
    train_group.add_argument('--batch_size', type=int, default=64,
                             help='Training batch size')
    train_group.add_argument('--cos_lr', action='store_true', default=False,
                             help='Use cosine annealing LR scheduler')
    train_group.add_argument('--optim', type=str, default='sgd',
                             choices=['sgd', 'adam'],
                             help='Optimizer')
    train_group.add_argument('--accumulation_steps', type=int, default=1,
                             help='Gradient accumulation steps')
    train_group.add_argument('--train_subset_size', type=float, default=1.0,
                             help='Fraction of calibration data per epoch')

    # ==== Weight Quantization (for INT4) ====
    wq_group = parser.add_argument_group('Weight Quantization (INT4)')
    wq_group.add_argument('--w_bits', type=int, default=4,
                          help='Weight quantization bits')
    wq_group.add_argument('--w_groupsize', type=int, default=-1,
                          help='Weight quantization group size (-1 for per-channel)')
    wq_group.add_argument('--w_asym', action='store_true', default=False,
                          help='Asymmetric weight quantization')
    wq_group.add_argument('--w_rtn', action='store_true', default=False,
                          help='Use RTN instead of GPTQ for weight quantization')
    wq_group.add_argument('--w_clip', action='store_true', default=False,
                          help='Enable weight clipping during quantization')
    wq_group.add_argument('--w_static_groups', action='store_true', default=False,
                          help='Static grouping for weight quantization')
    wq_group.add_argument('--percdamp', type=float, default=0.01,
                          help='GPTQ damping percentage')
    wq_group.add_argument('--act_order', action='store_true', default=False,
                          help='Activation order in GPTQ')

    # ==== Activation Quantization ====
    aq_group = parser.add_argument_group('Activation Quantization')
    aq_group.add_argument('--a_bits', type=int, default=4,
                          help='Activation quantization bits')
    aq_group.add_argument('--a_groupsize', type=int, default=-1,
                          help='Activation quantization group size')
    aq_group.add_argument('--a_asym', action='store_true', default=False,
                          help='Asymmetric activation quantization')
    aq_group.add_argument('--a_clip_ratio', type=float, default=1.0,
                          help='Activation clipping ratio')
    aq_group.add_argument('--a_residual', action='store_true', default=False,
                          help='Residual activation quantization')

    # ==== KV-Cache Quantization ====
    kv_group = parser.add_argument_group('KV-Cache Quantization')
    kv_group.add_argument('--k_bits', type=int, default=16,
                          help='K-cache quantization bits')
    kv_group.add_argument('--k_groupsize', type=int, default=-1)
    kv_group.add_argument('--k_asym', action='store_true', default=False)
    kv_group.add_argument('--k_clip_ratio', type=float, default=1.0)
    kv_group.add_argument('--v_bits', type=int, default=16,
                          help='V-cache quantization bits')
    kv_group.add_argument('--v_groupsize', type=int, default=-1)
    kv_group.add_argument('--v_asym', action='store_true', default=False)
    kv_group.add_argument('--v_clip_ratio', type=float, default=1.0)

    # ==== Calibration ====
    cal_group = parser.add_argument_group('Calibration')
    cal_group.add_argument('--cal_dataset', type=str, default='wikitext2',
                           choices=['wikitext2', 'c4', 'ptb'],
                           help='Calibration dataset')
    cal_group.add_argument('--nsamples', type=int, default=128,
                           help='Number of calibration samples')
    cal_group.add_argument('--seqlen', type=int, default=2048,
                           help='Sequence length')

    # ==== Evaluation ====
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--ppl_eval', action='store_true', default=True,
                            help='Evaluate perplexity')
    eval_group.add_argument('--no_ppl_eval', action='store_false', dest='ppl_eval')
    eval_group.add_argument('--ppl_eval_dataset', type=str, nargs='+',
                            default=['wikitext2'],
                            choices=['wikitext2', 'ptb', 'c4'],
                            help='Evaluation datasets')
    eval_group.add_argument('--ppl_eval_batch_size', type=int, default=1)

    # ==== Output ====
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--output_dir', type=str, default='./dartquant_v2_output',
                           help='Output directory for results and saved rotations')
    out_group.add_argument('--save_qmodel_path', type=str, default=None,
                           help='Path to save quantized model')
    out_group.add_argument('--save_rotations', action='store_true', default=True,
                           help='Save trained rotation matrices')
    out_group.add_argument('--fp32_had', action='store_true', default=False,
                           help='Apply Hadamard rotation in FP32')

    # ==== Misc ====
    misc_group = parser.add_argument_group('Misc')
    misc_group.add_argument('--wandb', action='store_true', default=False)
    misc_group.add_argument('--wandb_project', type=str, default=None)
    misc_group.add_argument('--wandb_id', type=str, default=None)
    misc_group.add_argument('--distribute', action='store_true', default=False,
                            help='Distribute model across multiple GPUs')

    return parser
