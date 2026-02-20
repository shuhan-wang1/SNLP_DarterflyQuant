import argparse
import pprint
import os
from datetime import datetime
import logging

from utils import supported_models, supported_datasets, config_logging


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Model to load;', choices=supported_models)
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--hf_token', type=str, default=None)

    # Rotation Arguments
    parser.add_argument('--fuse_norm', action=argparse.BooleanOptionalAction, default=True,
                        help='Fuse the normalization layer with the linear layer.')
    parser.add_argument('--smooth', type=str, default=None,
                        help='Smooth the rotation matrix.')
    parser.add_argument('--use_r1', action=argparse.BooleanOptionalAction, default=True,
                        help='''Use R1 for rotate attention, up-projection and gate-projection inputs.''')
    parser.add_argument('--r1_path', type=str, default=None,
                        help='''Path to the R1 rotation matrix. Deafult is None.
                        If not specified, R1 will generated as "rotate_mode".''')
    parser.add_argument('--use_r2', type=str, default='offline',
                        choices=['offline', 'online', 'none'],
                        help='''Use R2 for rotate out-projection inputs.''')
    parser.add_argument('--r2_path', type=str, default=None,
                        help='''Path to the R2 rotation matrix. Deafult is None.
                        If not specified, R2 will generated as "rotate_mode".''')
    parser.add_argument('--use_r3', action=argparse.BooleanOptionalAction, default=True,
                        help='''Use R3 for rotate Q/K Online.''')
    parser.add_argument('--use_r4', action=argparse.BooleanOptionalAction, default=True,
                        help='''Use R4 for rotate down-projection inputs Online.''')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    # parser.add_argument('--rotation_seed', type=int, default=-1,
    #                     help='Random Seed for generating random matrix!!')
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)''')
    parser.add_argument('--a_groupsize', type=int, default=-1,
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric Activation quantization (default: False)')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--a_residual', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether use residual quant for activation quantization (default: False)')

    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16,
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1,
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_static_groups', action=argparse.BooleanOptionalAction, default=False,
                        help='''Static Grouping for weight quantization.''')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')

    # General Quantization Arguments
    parser.add_argument('--w_bits_down_proj', type=int, default=None,
                        help='''Use special weight quantization bit width for Down Projection!
                        Default: w_bits.''')
    parser.add_argument('--a_bits_down_proj', type=int, default=None,
                        help='''Use special activation quantization bit width for Down Projection!
                        Default: a_bits.''')
    parser.add_argument('--o_per_head', action=argparse.BooleanOptionalAction, default=False,
                        help='Per-head quantization for out-projection')
    # parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
    #                     help='''Use INT8 for Down Projection! If this set,
    #                     both weights and activations of this layer will be in INT8''')

    # KV-Cache Quantization Arguments
    parser.add_argument('--v_bits', type=int, default=16,
                        help='''Number of bits for V-cache quantization.
                        Note that quantizing the V-cache does not need any other rotation''')
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric V-cache quantization')
    parser.add_argument('--v_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for v-cache quantization. new_max = max * clip_ratio')

    parser.add_argument('--k_bits', type=int, default=16,
                        help='''Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries''')
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric K-cache quantization')
    parser.add_argument('--k_pre_rope', action=argparse.BooleanOptionalAction, default=False,
                        help='Pre-RoPE quantization for K-cache (not Supported yet!)')
    parser.add_argument('--k_clip_ratio', type=float, default=1.0,
                        help='Clip ratio for k-cache quantization. new_max = max * clip_ratio')

    # Save/Load Quantized Model Arguments
    parser.add_argument('--load_qmodel_path', type=str, default=None,
                        help='Load the quantized model from the specified path!')
    parser.add_argument('--save_qmodel_path', type=str, default=None,
                        help='Save the quantized model to the specified path!')

    # WandB Arguments
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)

    # Experiments Arguments
    parser.add_argument('--save_name', type=str, default=None,
                        help='''The path to save experiment data,
                        including quantized models, dumped layer inputs, etc.
                        The data will be saved in experiments/[model]/save_name.
                        Default: [datetime].''')
    parser.add_argument('--log_to_console', action=argparse.BooleanOptionalAction, default=True,
                        help='Log to console')
    # Capture Layer Input/Output Arguments
    parser.add_argument('--capture_layer_io', action=argparse.BooleanOptionalAction, default=False,
                        help='Capture the input and output of the specified decoder layer and dump into a file')
    parser.add_argument('--layer_idx', type=int, default=10, help='Which decoder layer to capture')

    # PPL Eval Arguments
    parser.add_argument("--ppl_eval", action="store_true", help="Evaluate the model PPL")
    parser.add_argument('--ppl_eval_dataset', type=str, nargs='+', default=['wikitext2', 'ptb', 'c4'],
                        help='Dataset for Evaluation (default: wikitext2)', choices=supported_datasets,)
    parser.add_argument('--ppl_eval_batch_size', type=int, default=1,
                        help='Batch-size for PPL evaluation (default:1)')

    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy",
                 "arc_challenge", "winogrande", "lambada_openai",
                 "social_iqa", "openbookqa", "mmlu"],  # boolq
    )
    parser.add_argument('--lm_eval_batch_size', type=str, default='32',
                        help='Batch size for evaluating with lm eval harness.')
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Distribute the model on multiple GPUs for evaluation.",
    )

    # 权重微调参数:
    parser.add_argument('--w_ft', action=argparse.BooleanOptionalAction, default=False,
                        help='Whether to fine-tune weights to adapt to quantized activations(default: False).')
    parser.add_argument('--ft_percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')

    args = parser.parse_args()

    if args.lm_eval:
        from lm_eval.tasks import TaskManager   # lm_eval==0.4.3
        task_manager = TaskManager()
        task_names = task_manager.match_tasks(args.tasks)
        for task in [task for task in args.tasks if task not in task_names]:
            raise ValueError(f"Invalid task: {task}")

    if args.save_name is None:
        args.save_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    setattr(args, 'save_path',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', args.model, args.save_name))

    os.makedirs(args.save_path, exist_ok=True)

    config_logging(os.path.join(args.save_path, f'{args.save_name}.txt'),
                   to_console=args.log_to_console)

    # assert args.a_groupsize == args.w_groupsize, 'a_groupsize should be the same as w_groupsize!'
    assert args.k_pre_rope == False, 'Pre-RoPE quantization is not supported yet!'

    if args.model == 'facebook/opt-125m' or args.model == 'facebook/opt-1.3b':
        logging.warning('Warning: OPT-125M/1.3B is only for debugging purposes!!')

    if args.wandb:
        assert args.wandb_id is not None and args.wandb_project is not None, 'WandB ID/project is not provided!'

    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)
    return args
