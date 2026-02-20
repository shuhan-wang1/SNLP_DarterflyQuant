import torch
import random
import numpy as np
import os
import atexit
import logging
from tqdm import tqdm


from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

supported_models = [
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Llama-2-70b-hf',
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Meta-Llama-3-70B',
    'meta-llama/Llama-3.1-70B',
    'facebook/opt-125m'
]
supported_datasets = ['wikitext2', 'ptb', 'c4']

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def llama_down_proj_groupsize(model, groupsize):

    assert groupsize > 1, 'groupsize should be greater than 1!'

    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert groupsize * group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert down_proj_groupsize * group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


# Dump the log both to console and a log file.
def config_logging(log_file,
                   levels_to_log={logging.INFO, logging.ERROR},
                   to_console=True):
    # 确保路径存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 清除已有 handlers，避免冲突
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 定义日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 自定义过滤器
    class SpecificLevelFilter(logging.Filter):
        def filter(self, record):
            return record.levelno in levels_to_log

    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.addFilter(SpecificLevelFilter())  # 添加过滤器

    # 控制台处理器
    handlers = [file_handler]
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(SpecificLevelFilter())  # 添加过滤器
        handlers.append(console_handler)

    # 配置 logging
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    # 确保缓冲区在程序退出时写入
    atexit.register(logging.shutdown)


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2."""
    no_split_module_classes = ['LlamaDecoderLayer']
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    cleanup_memory()


import torch
import os
import logging


def save_model_in_parts(model, save_qmodel_path, prefix='model_part', num_digits=5, target_file_size=10 * 1024**3):
    """
    将模型按文件大小（例如 10GB）分块保存为多个文件。
    :param model: 要保存的 PyTorch 模型
    :param save_qmodel_path: 模型保存路径
    :param target_file_size: 每个文件的目标大小（单位：字节，默认10GB）
    :param prefix: 文件名前缀（默认 'model_part'）
    :param num_digits: 文件序号的位数（默认 5 位数）
    """
    state_dict = model.state_dict()

    # 计算模型每个参数的大小（根据参数的数据类型自动计算）
    total_size = 0
    for param in state_dict.values():
        param_size = param.element_size() * param.numel()  # 获取参数的大小（单位字节）
        total_size += param_size

    # 计算分块数量
    num_parts = (total_size + target_file_size - 1) // target_file_size  # 向上取整

    logging.info(
        f"模型总大小: {total_size / (1024**3):.2f} GB, 分为 {num_parts} 部分，每个部分约 {target_file_size / (1024**3):.2f} GB")

    # 分块保存模型
    idx = 0
    current_part_size = 0  # 当前分块的实际字节大小
    part = {}

    for name, param in state_dict.items():
        param_size = param.element_size() * param.numel()  # 计算当前参数的字节数
        current_part_size += param_size  # 累加当前分块的大小
        part[name] = param  # 添加当前的参数到部分分块中

        # 如果当前块的大小超过目标大小，保存并开始新的块
        if current_part_size >= target_file_size:
            # 格式化文件名，确保序号是 5 位数
            part_filename = f"{prefix}_{str(idx).zfill(num_digits)}.pth"
            torch.save(part, os.path.join(save_qmodel_path, part_filename))
            logging.info(f"保存了模型的第 {idx + 1} 部分：{part_filename}，共 {num_parts} 部分。")
            part = {}  # 清空当前部分，开始下一个分块
            current_part_size = 0  # 重置当前分块的大小
            idx += 1

    # 最后一块
    if part:
        part_filename = f"{prefix}_{str(idx).zfill(num_digits)}.pth"
        torch.save(part, os.path.join(save_qmodel_path, part_filename))
        logging.info(f"保存了模型的第 {idx + 1} 部分：{part_filename}，共 {num_parts} 部分。")

    logging.info("模型分块保存完成。")


def load_model_in_parts(model, folder_path):
    """
    将模型的多个部分加载并逐块赋值给模型。
    :param model: 要加载的 PyTorch 模型
    :param folder_path: 存储分块模型文件的文件夹路径
    """
    # 获取文件夹中的所有 .pth 文件，按名称排序（确保加载顺序正确）
    model_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pth')])

    # 逐个加载分块
    with tqdm(total=len(model_files), desc="Loading Model Parts", unit="part") as pbar:
        for file_name in model_files:
            part = torch.load(os.path.join(folder_path, file_name), map_location='cpu')  # 加载分块

            model.load_state_dict(part, strict=False)  # 更新模型的参数（实时赋值）

            del part  # 释放已加载分块的内存
            pbar.update(1)  # 更新进度条

    logging.info("模型加载完成。")
