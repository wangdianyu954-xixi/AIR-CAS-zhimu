#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run end-to-end inference for RGB, SAR, multispectral, and hyperspectral inputs.

1. RGB: 3 channels
2. SAR: 1 channel
3. Multispectral: 12 channels by default
4. Hyperspectral: 218 channels by default

Features:
- Uses only local model files and does not access the Hugging Face Hub.
- Explicitly imports the local modeling_qwen3_vl.py implementation.
- Uses the local image_processing_qwen2_vl.py implementation.
- Converts non-RGB arrays to channels-first layout.
- Validates processor channel metadata.
- Validates the flattened patch dimension.
- Verifies the active patch embedding branch with forward hooks.
- Prints the generated response and peak GPU memory usage.

Example:
python test_multimodal_qwen3vl_local.py     --model-dir /path/to/model     --input /path/to/sar.npy     --modality sar     --prompt "Describe the main objects in this SAR image."
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Callable
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch
import transformers
from PIL import Image
DEFAULT_CHANNELS = {'rgb': 3, 'sar': 1, 'multispectral': 12, 'hyperspectral': 218}
DEFAULT_PROMPTS = {'rgb': 'Describe the main land-cover types and objects in this remote-sensing image.', 'sar': 'Describe the main scene and objects in this SAR image.', 'multispectral': 'Identify the main land-cover types in this multispectral image.', 'hyperspectral': 'Identify the main land-cover types in this hyperspectral image.'}
EXPECTED_BRANCH_KEYS = {'rgb': 'patch_embed.proj', 'sar': 'extra_patch_embeds.ch_1.proj', 'multispectral': 'extra_patch_embeds.ch_12.proj', 'hyperspectral': 'extra_patch_embeds.ch_218.proj'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test local Qwen3-VL inference for RGB, SAR, multispectral, and hyperspectral inputs.')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory containing model weights and local model code.')
    parser.add_argument('--input', type=str, required=True, help='Input file path.')
    parser.add_argument('--modality', type=str, required=True, choices=list(DEFAULT_CHANNELS), help='Input modality.')
    parser.add_argument('--expected-channels', type=int, default=None, help='Override the default channel count: RGB=3, SAR=1, multispectral=12, hyperspectral=218.')
    parser.add_argument('--input-layout', type=str, choices=['auto', 'chw', 'hwc'], default='auto', help='Layout of a non-RGB array; auto infers the channel axis from expected_channels.')
    parser.add_argument('--npz-key', type=str, default=None, help='Array key for NPZ input; omit only when the file contains one array.')
    parser.add_argument('--mat-key', type=str, default=None, help='Variable name for MAT input.')
    parser.add_argument('--prompt', type=str, default=None, help='Inference prompt; uses a modality-specific default when omitted.')
    parser.add_argument('--max-new-tokens', type=int, default=128, help='Maximum number of generated tokens.')
    parser.add_argument('--attn', type=str, choices=['sdpa', 'eager'], default='sdpa', help='Attention implementation.')
    parser.add_argument('--dtype', type=str, choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto', help='Model inference dtype.')
    parser.add_argument('--no-strict-branch-check', action='store_true', help='Warn instead of failing when the patch embedding branch does not match.')
    return parser.parse_args()

def print_section(title: str) -> None:
    print('\n' + '=' * 88)
    print(title)
    print('=' * 88)

def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def print_environment() -> None:
    print_section('Environment')
    print(f'Python       : {sys.version.split()[0]}')
    print(f'PyTorch      : {torch.__version__}')
    print(f'Transformers : {transformers.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f'GPU {idx}       : {props.name}, {props.total_memory / 1024 ** 3:.2f} GB')

def validate_model_dir(model_dir: Path) -> dict[str, Any]:
    print_section('Model directory')
    required = ['config.json', 'model.safetensors.index.json', 'modeling_qwen3_vl.py', 'configuration_qwen3_vl.py', 'image_processing_qwen2_vl.py', 'tokenizer_config.json', 'preprocessor_config.json']
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError('Required model files are missing:\n' + '\n'.join((f'  - {name}' for name in missing)))
    config = load_json(model_dir / 'config.json')
    index_data = load_json(model_dir / 'model.safetensors.index.json')
    shard_names = sorted(set(index_data.get('weight_map', {}).values()))
    if not shard_names:
        raise RuntimeError('No weight shards were found in the safetensors index.')
    missing_shards = [shard for shard in shard_names if not (model_dir / shard).exists()]
    if missing_shards:
        raise FileNotFoundError('The following weight shards are missing:\n' + '\n'.join((f'  - {name}' for name in missing_shards)))
    print(f'Model directory: {model_dir}')
    print(f"model_type    : {config.get('model_type')}")
    print(f"architectures : {config.get('architectures')}")
    print(f"auto_map      : {config.get('auto_map')}")
    print(f'Weight shards: {len(shard_names)}')
    for shard in shard_names:
        path = model_dir / shard
        print(f'  {shard}: {path.stat().st_size / 1024 ** 3:.2f} GB')
    return config

def create_local_package(model_dir: Path) -> str:
    """
    Register the model directory as a temporary Python package so local relative imports work.
    """
    package_name = '_rsmllm_local_model'
    if package_name in sys.modules:
        return package_name
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    package.__package__ = package_name
    package.__file__ = str(model_dir / '__init__.py')
    sys.modules[package_name] = package
    return package_name

def select_dtype(name: str) -> torch.dtype:
    if name == 'bf16':
        return torch.bfloat16
    if name == 'fp16':
        return torch.float16
    if name == 'fp32':
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def load_local_model(model_dir: Path, config: dict[str, Any], dtype: torch.dtype, attn_implementation: str) -> Any:
    print_section('Loading local model code')
    package_name = create_local_package(model_dir)
    modeling_module = importlib.import_module(f'{package_name}.modeling_qwen3_vl')
    architectures = config.get('architectures') or ['Qwen3VLForConditionalGeneration']
    class_name = architectures[0]
    if not hasattr(modeling_module, class_name):
        candidates = [name for name in dir(modeling_module) if 'ForConditionalGeneration' in name]
        raise AttributeError(f'The requested class is not defined in local modeling_qwen3_vl.py: {class_name}.\nCandidate model classes: {candidates}')
    model_class = getattr(modeling_module, class_name)
    print(f'Model class: {model_class.__name__}')
    print(f'Source: {model_class.__module__}')
    print(f'dtype : {dtype}')
    common_kwargs = dict(local_files_only=True, device_map='auto', low_cpu_mem_usage=True, attn_implementation=attn_implementation)
    try:
        model = model_class.from_pretrained(str(model_dir), dtype=dtype, **common_kwargs)
    except TypeError as exc:
        if 'dtype' not in str(exc):
            raise
        model = model_class.from_pretrained(str(model_dir), torch_dtype=dtype, **common_kwargs)
    model.eval()
    print(f'Loaded model source: {model.__class__.__module__}')
    if model.__class__.__module__ != f'{package_name}.modeling_qwen3_vl':
        raise RuntimeError('The loaded model class does not come from local modeling_qwen3_vl.py.')
    total_params = sum((p.numel() for p in model.parameters()))
    print(f'Total parameters: {total_params / 1000000000.0:.3f} B')
    return model

def load_processor_with_local_image_code(model_dir: Path) -> Any:
    print_section('Loading processor and local image processor')
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True, use_fast=False)
    package_name = create_local_package(model_dir)
    image_module = importlib.import_module(f'{package_name}.image_processing_qwen2_vl')
    if hasattr(image_module, 'Qwen3VLImageProcessor'):
        image_processor_class = image_module.Qwen3VLImageProcessor
    elif hasattr(image_module, 'Qwen2VLImageProcessor'):
        image_processor_class = image_module.Qwen2VLImageProcessor
    else:
        raise AttributeError('No supported image processor class was found in local image_processing_qwen2_vl.py: Qwen3VLImageProcessor or Qwen2VLImageProcessor.')
    local_image_processor = image_processor_class.from_pretrained(str(model_dir), local_files_only=True)
    if not hasattr(processor, 'image_processor'):
        raise AttributeError('AutoProcessor does not expose an image_processor attribute.')
    processor.image_processor = local_image_processor
    print(f'Processor class: {processor.__class__.__name__}')
    print(f'Processor source: {processor.__class__.__module__}')
    print(f'Image processor class: {processor.image_processor.__class__.__name__}')
    print(f'Image processor source: {processor.image_processor.__class__.__module__}')
    print(f'patch_size        :{processor.image_processor.patch_size}')
    print(f'temporal_patch_size:{processor.image_processor.temporal_patch_size}')
    print(f'merge_size        :{processor.image_processor.merge_size}')
    print(f"hyperspectral_divisor:{getattr(processor.image_processor, 'hyperspectral_divisor', None)}")
    expected_source = f'{package_name}.image_processing_qwen2_vl'
    if processor.image_processor.__class__.__module__ != expected_source:
        raise RuntimeError('The active image processor does not come from local image_processing_qwen2_vl.py.')
    return processor

def load_npz(path: Path, key: str | None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        if key is None:
            if len(keys) != 1:
                raise ValueError(f'The NPZ file contains multiple arrays {keys}; specify one with --npz-key.')
            key = keys[0]
        if key not in data:
            raise KeyError(f'The NPZ file does not contain key {key!r}; available keys: {keys}')
        return np.asarray(data[key])

def load_mat(path: Path, key: str | None) -> np.ndarray:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError('Reading MAT files requires scipy.') from exc
    data = loadmat(path)
    candidates = {k: v for k, v in data.items() if not k.startswith('__') and isinstance(v, np.ndarray)}
    if key is None:
        if len(candidates) != 1:
            raise ValueError(f'Candidate arrays in the MAT file: {list(candidates)}; specify one with --mat-key.')
        key = next(iter(candidates))
    if key not in candidates:
        raise KeyError(f'The MAT file does not contain array {key!r}; candidate arrays: {list(candidates)}')
    return np.asarray(candidates[key])

def load_non_rgb_array(input_path: Path, npz_key: str | None, mat_key: str | None) -> np.ndarray:
    suffix = input_path.suffix.lower()
    if suffix == '.npy':
        return np.asarray(np.load(input_path, allow_pickle=False))
    if suffix == '.npz':
        return load_npz(input_path, npz_key)
    if suffix in {'.tif', '.tiff'}:
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError('Reading TIFF files requires tifffile.') from exc
        return np.asarray(tifffile.imread(input_path))
    if suffix == '.mat':
        return load_mat(input_path, mat_key)
    with Image.open(input_path) as image:
        return np.asarray(image)

def convert_array_to_chw(array: np.ndarray, expected_channels: int, input_layout: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        if expected_channels != 1:
            raise ValueError(f'The input is a 2D array {array.shape}, but the expected channel count is {expected_channels}.')
        return array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f'Only 2D or 3D single-image arrays are supported; got shape={array.shape}.')
    if input_layout == 'chw':
        chw = array
    elif input_layout == 'hwc':
        chw = np.moveaxis(array, -1, 0)
    else:
        candidate_axes = [axis for axis, size in enumerate(array.shape) if size == expected_channels]
        if len(candidate_axes) == 0:
            raise ValueError(f'Unable to find the expected channel axis in shape={array.shape}; expected channels={expected_channels}. Specify --input-layout or verify --expected-channels.')
        if len(candidate_axes) > 1:
            if 0 in candidate_axes:
                channel_axis = 0
            elif 2 in candidate_axes:
                channel_axis = 2
            else:
                raise ValueError(f'shape={array.shape} contains multiple axes matching {expected_channels}; specify --input-layout explicitly.')
        else:
            channel_axis = candidate_axes[0]
        chw = np.moveaxis(array, channel_axis, 0)
    if chw.shape[0] != expected_channels:
        raise ValueError(f'The channel count after CHW conversion is {chw.shape[0]}; expected {expected_channels}; shape={chw.shape}.')
    return np.ascontiguousarray(chw)

def describe_array(name: str, array: np.ndarray) -> None:
    print(f'{name} shape: {array.shape}')
    print(f'{name} dtype: {array.dtype}')
    print(f'{name} min: {float(np.nanmin(array))}')
    print(f'{name} max: {float(np.nanmax(array))}')
    print(f'{name} mean: {float(np.nanmean(array))}')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} contains NaN or Inf.')

def load_visual_input(input_path: Path, modality: str, expected_channels: int, input_layout: str, npz_key: str | None, mat_key: str | None) -> Image.Image | np.ndarray:
    print_section('Loading input')
    print(f'Input path: {input_path}')
    print(f'Input modality.:{modality}')
    print(f'Expected channels: {expected_channels}')
    if modality == 'rgb':
        image = Image.open(input_path).convert('RGB')
        print(f'RGB size: {image.size}')
        print('RGB mode: RGB')
        return image
    array = load_non_rgb_array(input_path, npz_key, mat_key)
    describe_array('Raw array', array)
    chw = convert_array_to_chw(array=array, expected_channels=expected_channels, input_layout=input_layout)
    describe_array('CHW array', chw)
    return chw

def validate_processor_outputs(inputs: Any, processor: Any, expected_channels: int) -> None:
    print_section('Validating processor output')
    for key, value in inputs.items():
        if hasattr(value, 'shape'):
            print(f"{key:20s}: shape={tuple(value.shape)}, dtype={getattr(value, 'dtype', None)}")
        else:
            print(f'{key:20s}: {type(value).__name__}')
    if 'num_channels' not in inputs:
        raise RuntimeError('Processor output does not contain num_channels.')
    actual_channels = [int(v) for v in inputs['num_channels'].flatten().tolist()]
    print(f'Processor num_channels: {actual_channels}')
    if any((v != expected_channels for v in actual_channels)):
        raise RuntimeError(f'Channel mismatch: expected {expected_channels}, got {actual_channels}.')
    if 'pixel_values' not in inputs:
        raise RuntimeError('Processor output does not contain pixel_values.')
    pixel_values = inputs['pixel_values']
    if not hasattr(pixel_values, 'shape') or len(pixel_values.shape) != 2:
        raise RuntimeError(f"pixel_values must be a 2D tensor; got {type(pixel_values)} / {getattr(pixel_values, 'shape', None)}")
    patch_size = int(processor.image_processor.patch_size)
    temporal_patch_size = int(processor.image_processor.temporal_patch_size)
    expected_flat_dim = expected_channels * temporal_patch_size * patch_size * patch_size
    actual_flat_dim = int(pixel_values.shape[-1])
    print(f'Expected flattened patch dimension: {expected_flat_dim}')
    print(f'Actual flattened patch dimension: {actual_flat_dim}')
    if actual_flat_dim != expected_flat_dim:
        raise RuntimeError(f'Flattened patch dimension mismatch: expected {expected_flat_dim}, got {actual_flat_dim}.')
    print('Processor channel count and patch dimension checks passed.')

def register_patch_branch_hooks(model: Any) -> tuple[list[str], list[Any], list[str]]:
    """
    Register hooks for the primary and additional patch embedding branches.
    """
    candidate_names: list[str] = []
    called_names: list[str] = []
    handles: list[Any] = []

    def make_hook(name: str) -> Callable:

        def hook(_module: Any, _inputs: Any, _output: Any) -> None:
            called_names.append(name)
        return hook
    for name, module in model.named_modules():
        is_main_proj = name.endswith('visual.patch_embed.proj')
        is_extra_proj = 'visual.patch_embed.extra_patch_embeds.' in name and name.endswith('.proj')
        if is_main_proj or is_extra_proj:
            candidate_names.append(name)
            handles.append(module.register_forward_hook(make_hook(name)))
    print_section('Available patch embedding branches')
    if candidate_names:
        for name in candidate_names:
            print(f'  - {name}')
    else:
        print('No patch embedding module was found.')
    return (called_names, handles, candidate_names)

def validate_called_branch(called_names: list[str], candidate_names: list[str], modality: str, expected_channels: int, strict: bool) -> None:
    print_section('Validating patch embedding branch')
    unique_called = list(dict.fromkeys(called_names))
    if unique_called:
        print('Observed calls:')
        for name in unique_called:
            print(f'  - {name}')
    else:
        print('No patch embedding call was captured.')
        if candidate_names:
            message = 'Patch embedding candidates were found, but no hook was triggered.'
            if strict:
                raise RuntimeError(message)
            print(f'[Warning] {message}')
        return
    if modality == 'rgb':
        matched = any((name.endswith('visual.patch_embed.proj') and 'extra_patch_embeds' not in name for name in unique_called))
        expected_text = 'primary RGB patch_embed.proj'
    else:
        key = f'extra_patch_embeds.ch_{expected_channels}.proj'
        matched = any((key in name for name in unique_called))
        expected_text = key
    if not matched:
        message = f'Expected {expected_text}, but observed {unique_called}.'
        if strict:
            raise RuntimeError(message)
        print(f'[Warning] {message}')
    else:
        print(f'Confirmed branch: {expected_text}')

def move_inputs_to_model_device(inputs: Any, model: Any) -> Any:
    try:
        target_device = model.device
    except Exception:
        target_device = next(model.parameters()).device
    print(f'Input device: {target_device}')
    if hasattr(inputs, 'to'):
        return inputs.to(target_device)
    return {key: value.to(target_device) if hasattr(value, 'to') else value for key, value in inputs.items()}

def run_inference(model: Any, processor: Any, visual_input: Image.Image | np.ndarray, modality: str, expected_channels: int, prompt: str, max_new_tokens: int, strict_branch_check: bool) -> str:
    print_section('Preparing multimodal input')
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': visual_input}, {'type': 'text', 'text': prompt}]}]
    print(f'Prompt: {prompt}')
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
    validate_processor_outputs(inputs=inputs, processor=processor, expected_channels=expected_channels)
    called_names, handles, candidate_names = register_patch_branch_hooks(model)
    inputs = move_inputs_to_model_device(inputs, model)
    input_length = int(inputs['input_ids'].shape[1])
    print_section('Generating response')
    try:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    validate_called_branch(called_names=called_names, candidate_names=candidate_names, modality=modality, expected_channels=expected_channels, strict=strict_branch_check)
    generated_ids = generated_ids[:, input_length:]
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer

def print_gpu_memory() -> None:
    if not torch.cuda.is_available():
        return
    print_section('GPU memory')
    for idx in range(torch.cuda.device_count()):
        print(f'GPU {idx}: allocated={torch.cuda.memory_allocated(idx) / 1024 ** 3:.2f} GB, reserved={torch.cuda.memory_reserved(idx) / 1024 ** 3:.2f} GB, peak={torch.cuda.max_memory_allocated(idx) / 1024 ** 3:.2f} GB')

def print_supported_patch_branches(model: Any) -> None:
    print_section('Multichannel patch embedding')
    patch_embed = None
    for path in (('model', 'visual', 'patch_embed'), ('visual', 'patch_embed')):
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            patch_embed = obj
            break
        except AttributeError:
            continue
    if patch_embed is None:
        print('patch_embed was not found.')
        return
    print(f'PatchEmbed class: {patch_embed.__class__.__name__}')
    extra = getattr(patch_embed, 'extra_patch_embeds', None)
    if extra is None:
        print('extra_patch_embeds was not found.')
        return
    try:
        keys = list(extra.keys())
    except Exception:
        keys = []
    print(f'Additional channel branches: {keys}')

def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(f'Model directory does not exist: {model_dir}')
    if not input_path.is_file():
        raise FileNotFoundError(f'Input file does not exist: {input_path}')
    expected_channels = args.expected_channels if args.expected_channels is not None else DEFAULT_CHANNELS[args.modality]
    prompt = args.prompt or DEFAULT_PROMPTS[args.modality]
    print_environment()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. CPU inference is not recommended for this model.')
    for idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(idx)
    config = validate_model_dir(model_dir)
    dtype = select_dtype(args.dtype)
    model = load_local_model(model_dir=model_dir, config=config, dtype=dtype, attn_implementation=args.attn)
    print_supported_patch_branches(model)
    processor = load_processor_with_local_image_code(model_dir)
    visual_input = load_visual_input(input_path=input_path, modality=args.modality, expected_channels=expected_channels, input_layout=args.input_layout, npz_key=args.npz_key, mat_key=args.mat_key)
    answer = run_inference(model=model, processor=processor, visual_input=visual_input, modality=args.modality, expected_channels=expected_channels, prompt=prompt, max_new_tokens=args.max_new_tokens, strict_branch_check=not args.no_strict_branch_check)
    print_section('Model response')
    print(answer)
    print_gpu_memory()
    print_section('Result')
    print(f'{args.modality} modality test completed: local model code, local image processor, {expected_channels}-channel preprocessing, patch embedding validation, and generation succeeded.')
if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('\n' + '!' * 88)
        print('Inference failed:')
        print('!' * 88)
        traceback.print_exc()
        sys.exit(1)
