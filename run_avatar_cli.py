import os
import time
import sys
import argparse
import traceback
import subprocess
from pathlib import Path
from PIL import Image

import torch
import gc
import numpy as np
import librosa
from tqdm import tqdm

from mmgp import offload, profile_type
from wan.modules.attention import get_supported_attention_modes
from wan.utils.utils import cache_video

import uuid, requests, pathlib

ACCESS_KEY = "17e23633-2a7a-4d29-9450be4d6c8e-e01f-45f4"
unique_key = uuid.uuid4().hex

# runpod-volume


# --- CÁC BIẾN TOÀN CỤC ---
bfloat16_supported = torch.cuda.get_device_capability()[0] >= 8

hunyuan_choices = [
    "ckpts/hunyuan_video_avatar_720_bf16.safetensors",
    "ckpts/hunyuan_video_avatar_720_quanto_bf16_int8.safetensors",
]
transformer_choices = hunyuan_choices

model_signatures = {
    "hunyuan_avatar": "hunyuan_video_avatar"
}


# -------------------------

def get_model_family(model_filename):
    if "hunyuan" in model_filename:
        return "hunyuan"
    raise Exception(f"Unknown model family for model'{model_filename}'")


def get_transformer_dtype(model_family, transformer_dtype_policy):
    if len(transformer_dtype_policy) == 0:
        return torch.bfloat16 if bfloat16_supported else torch.float16
    elif transformer_dtype_policy == "fp16":
        return torch.float16
    else:
        return torch.bfloat16


def get_model_filename(model_type, quantization="int8", dtype_policy=""):
    signature = model_signatures[model_type]
    choices = [name for name in transformer_choices if signature in name]
    if not choices:
        raise ValueError(f"No model found for type '{model_type}' with signature '{signature}'")

    if len(quantization) == 0:
        quantization = "bf16"

    model_family = get_model_family(choices[0])
    dtype = get_transformer_dtype(model_family, dtype_policy)

    sub_choices = [name for name in choices if quantization in name]
    if len(sub_choices) > 0:
        dtype_str = "fp16" if dtype == torch.float16 else "bf16"
        new_sub_choices = [name for name in sub_choices if dtype_str in name]
        raw_filename = new_sub_choices[0] if len(new_sub_choices) > 0 else sub_choices[0]
    else:
        raw_filename = choices[0]

    return raw_filename


def get_hunyuan_text_encoder_filename(text_encoder_quantization):
    if text_encoder_quantization == "int8":
        return "ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_quanto_int8.safetensors"
    else:
        return "ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_fp16.safetensors"


def load_hunyuan_model(model_filename, text_encoder_quantization, dtype, VAE_dtype, mixed_precision_transformer):
    from hyvideo.hunyuan import HunyuanVideoSampler
    from hyvideo.modules.models import get_linear_split_map
    print(f"Loading '{model_filename[-1]}' model...")
    hunyuan_model = HunyuanVideoSampler.from_pretrained(
        model_filepath=model_filename,
        text_encoder_filepath=get_hunyuan_text_encoder_filename(text_encoder_quantization),
        dtype=dtype,
        VAE_dtype=VAE_dtype,
        mixed_precision_transformer=mixed_precision_transformer
    )
    pipe = {"transformer": hunyuan_model.model, "text_encoder": hunyuan_model.text_encoder,
            "text_encoder_2": hunyuan_model.text_encoder_2, "vae": hunyuan_model.vae}
    if hunyuan_model.wav2vec is not None:
        pipe["wav2vec"] = hunyuan_model.wav2vec
    split_linear_modules_map = get_linear_split_map()
    hunyuan_model.model.split_linear_modules_map = split_linear_modules_map
    offload.split_linear_modules(hunyuan_model.model, split_linear_modules_map)
    return hunyuan_model, pipe


def initialize_model(args):
    """Tải và cấu hình mô hình Hunyuan Avatar với các tùy chọn nâng cao."""
    model_type = "hunyuan_avatar"
    transformer_filename = get_model_filename(model_type, args.quantization, args.dtype)
    text_encoder_quantization = args.quantization

    # print("\n--- Model Configuration ---")
    # print(f"Transformer: {transformer_filename}")
    # print(f"Quantization: {args.quantization}")
    # print(f"Data Type: {args.dtype or 'auto'}")
    # print(f"Attention: {args.attention}")
    # print(f"VAE Tiling: {args.vae_tiling_config}")
    # print("---------------------------")

    if not Path(transformer_filename).exists():
        print(f"ERROR: Model file not found at {transformer_filename}")
        sys.exit(1)

    offload.shared_state["_attention"] = args.attention
    model_filelist = [transformer_filename]
    transformer_dtype = get_transformer_dtype("hunyuan", args.dtype)
    VAE_dtype = torch.float16 if args.vae_precision == "16" else torch.float32

    wan_model, pipe = load_hunyuan_model(
        model_filelist,
        text_encoder_quantization=text_encoder_quantization,
        dtype=transformer_dtype,
        VAE_dtype=VAE_dtype,
        mixed_precision_transformer=(args.mixed_precision == "1")
    )
    wan_model._model_file_name = transformer_filename

    profile = args.profile
    print(f"Profile: {profile}")
    budgets = {}
    if profile == 2:
        budgets = {"text_encoder": 1, "*": "100%"}
    elif profile in (4, 5):
        budgets = {"transformer": 8000, "text_encoder": 100, "*": 3000}
    elif profile == 3:
        budgets = {"*": "70%"}

    if budgets:
        kwargs = {"budgets": budgets}
    else:
        kwargs = {}

    offload.profile(pipe, profile_no=profile, convertWeightsFloatTo=transformer_dtype, **kwargs)

    return wan_model


def run_inference(wan_model, args):
    """Chạy suy luận và tạo video."""
    torch.set_grad_enabled(False)

    image_path = Path(args.image).resolve()
    audio_path = Path(args.audio).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not image_path.is_file():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)

    if not audio_path.is_file():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    image_start = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image_start.size

    if orig_w <= orig_h:  # ảnh dọc hoặc vuông
        width = 480  # cạnh ngắn
        height = int(orig_h * 480 / orig_w)
    else:  # ảnh ngang
        height = 480
        width = int(orig_w * 480 / orig_h)
    image_start = image_start.resize((width, height), Image.Resampling.LANCZOS)
    image_refs = [image_start]

    # image_start = Image.open(image_path).convert('RGB')
    # image_refs = [image_start]
    #
    # width, height = 480, 640
    seed = args.seed if args.seed != -1 else np.random.randint(0, 999999999)
    fps = 25

    audio_duration = librosa.get_duration(path=str(audio_path))
    video_length = min(int(fps * audio_duration // 4) * 4 + 5, 401)

    # print("\n--- Generation Parameters ---")
    # print(f"Resolution: {width}x{height}")
    # print(f"Video Length: {video_length} frames")
    # print(f"Steps: {args.steps}")
    # print(f"Seed: {seed}")
    # print(f"Guidance Scale: {args.guidance_scale}")
    # print(f"Flow Shift: {args.flow_shift}")
    # print("-----------------------------")

    # <<< THAY ĐỔI: Sử dụng logic VAE Tiling từ wgp.py gốc
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
    vae_tile_size_config = wan_model.vae.get_VAE_tile_size(
        args.vae_tiling_config,
        device_mem_capacity,
        args.vae_precision == "32"
    )

    # Cấu hình TeaCache
    trans = wan_model.model
    if args.teacache > 0:
        trans.enable_teacache = True
        trans.teacache_multiplier = args.teacache
        trans.teacache_start_step = int(args.teacache_start_perc * args.steps / 100)
    else:
        trans.enable_teacache = False

    progress_bar = tqdm(total=args.steps, desc="Denoising")

    def callback(step_idx, latent, force_refresh, **kwargs):
        progress_bar.update(1)

    start_time = time.time()
    try:
        # <<< THAY ĐỔI: Đơn giản hóa lời gọi, chỉ truyền các tham số cần thiết
        samples = wan_model.generate(
            input_prompt=args.prompt,
            input_ref_images=image_refs,
            frame_num=video_length,
            height=height,
            width=width,
            sampling_steps=args.steps,
            guide_scale=args.guidance_scale,
            shift=args.flow_shift,
            seed=seed,
            callback=callback,
            audio_guide=str(audio_path),
            fps=fps,
            VAE_tile_size=vae_tile_size_config,
            model_filename=wan_model._model_file_name,
        )
    except Exception as e:
        print("\nAn error occurred during video generation:")
        traceback.print_exc()
        return
    finally:
        progress_bar.close()

    if samples is None:
        print(f"ERROR: Generation failed. The model returned: {samples}")
        return

    # samples = samples["x"].to("cpu")

    # Lưu kết quả
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = f"avatar_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    temp_video_path = output_dir / f"{base_filename}_temp.mp4"
    final_video_path = output_dir / f"{base_filename}.mp4"

    # print(f"\nSaving temporary video to {temp_video_path}...")
    cache_video(tensor=samples[None], save_file=str(temp_video_path), fps=fps, nrow=1, normalize=True,
                value_range=(-1, 1))

    # print(f"Combining video and audio into {final_video_path} using ffmpeg...")
    ffmpeg_command = ["ffmpeg", "-y", "-i", str(temp_video_path), "-i", str(audio_path),
                      "-c:v", "libx264", "-c:a", "aac", "-shortest",
                      "-loglevel", "warning", "-nostats", str(final_video_path)]
    try:
        subprocess.run(ffmpeg_command, check=True)
        os.remove(temp_video_path)
        end_time = time.time()
        print("\n--- Generation Complete! ---")
        print(f"Video saved to: {final_video_path}")
        print(f"Total time: {end_time - start_time:.2f} seconds.")
        LOCAL_FILE = pathlib.Path(final_video_path)
        url = f"https://storage.bunnycdn.com/zockto/video/{base_filename}.mp4"
        with LOCAL_FILE.open("rb") as f:
            r = requests.put(
                url,
                headers={
                    "AccessKey": ACCESS_KEY,
                    "Content-Type": "video/mp4"
                },
                data=f,  # stream theo chunk, không load hết vào RAM
                timeout=1200  # tăng timeout nếu file lớn
            )
        # print(r.status_code, r.text)
        print("CDN URL:", f"https://zockto.b-cdn.net/video/{base_filename}.mp4")


    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: ffmpeg command failed. Ensure ffmpeg is installed.")
        print(f"Video (no audio) is at: {temp_video_path}")


def main():
    parser = argparse.ArgumentParser(description="CLI for Hunyuan Video Avatar generation, mimicking wgp.py defaults.")

    # --- Input Arguments ---
    input_group = parser.add_argument_group('Input Arguments')
    input_group.add_argument("--image", type=str, required=True, help="Path to the input image.")
    input_group.add_argument("--audio", type=str, required=True, help="Path to the input audio.")
    input_group.add_argument("--prompt", type=str, required=True, help="Text prompt.")

    # --- Generation Arguments ---
    gen_group = parser.add_argument_group('Generation Arguments')
    gen_group.add_argument("--output_dir", type=str, default="outputs_cli", help="Output directory.")
    gen_group.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    gen_group.add_argument("--seed", type=int, default=-1, help="Seed for generation (-1 for random).")
    gen_group.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    gen_group.add_argument("--flow_shift", type=float, default=5.0, help="Flow shift scale.")

    # --- Performance Arguments ---
    perf_group = parser.add_argument_group('Performance Arguments')
    perf_group.add_argument("--quantization", type=str, default="int8", choices=["int8", "bf16"],
                            help="Model quantization.")
    perf_group.add_argument("--dtype", type=str, default="bf16", choices=["", "fp16", "bf16"],
                            help="Transformer data type (auto-detect).")
    perf_group.add_argument("--attention", type=str, default="sage2", choices=["sdpa", "sage", "sage2"],
                            help="Attention mechanism.")
    perf_group.add_argument("--profile", type=int, default=2, choices=[1, 2, 3, 4, 5],
                            help="Memory/VRAM usage profile (1-5).")
    perf_group.add_argument("--vae_tiling_config", type=int, default=1, choices=[0, 1, 2, 3],
                            help="VAE Tiling config (0:Auto, 1:Off, 2:8GB+, 3:6GB+).")
    perf_group.add_argument("--vae_precision", type=str, default="16", choices=["16", "32"],
                            help="VAE precision (16/32 bit).")
    perf_group.add_argument("--mixed_precision", type=str, default="0", choices=["0", "1"],
                            help="Transformer mixed precision (0:off, 1:on).")

    # --- TeaCache Arguments ---
    teacache_group = parser.add_argument_group('TeaCache Arguments')
    teacache_group.add_argument("--teacache", type=float, default=25,
                                help="Enable TeaCache with a multiplier (0 to disable).")
    teacache_group.add_argument("--teacache_start_perc", type=int, default=25,
                                help="Start TeaCache after this percentage of steps.")

    args = parser.parse_args()

    # --- Validation ---
    try:
        supported_attentions = get_supported_attention_modes()
        if args.attention not in supported_attentions:
            print(
                f"Warning: Attention '{args.attention}' not supported/installed. Supported: {supported_attentions}. Falling back to sdpa.")
            args.attention = 'sdpa'
    except Exception:
        print("Warning: Could not check supported attention modes. Assuming 'sdpa' is available.")
        args.attention = 'sdpa'

    wan_model = initialize_model(args)
    run_inference(wan_model, args)

    if offload.last_offload_obj is not None:
        offload.last_offload_obj.release()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()