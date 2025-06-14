import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

# Thư mục gốc để lưu trữ tất cả các mô hình, giống như trong wgp.py
ROOT_DIR = "ckpts"


def compute_list_from_path(filename: str) -> list[str]:
    """Trích xuất tên tệp từ một đường dẫn đầy đủ."""
    pos = filename.rfind("/")
    filename = filename[pos + 1:]
    return [filename]


def download_with_progress(repo_id, subfolder, filename, target_dir):
    """Sử dụng hf_hub_download với thanh tiến trình tùy chỉnh."""
    print(f"  - Tải tệp: {filename}")
    # hf_hub_download đã tích hợp sẵn thanh tiến trình của tqdm
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )


def process_files_def(repo_id: str, source_folder_list: list[str], file_list: list[list[str]]):
    """
    Tải xuống các tệp được chỉ định từ một kho lưu trữ Hugging Face.

    Args:
        repo_id: ID của kho lưu trữ trên Hugging Face.
        source_folder_list: Danh sách các thư mục con trong kho lưu trữ.
        file_list: Danh sách các danh sách tệp, tương ứng với mỗi thư mục con.
    """
    for source_folder, files_to_download in zip(source_folder_list, file_list):
        target_sub_dir = Path(ROOT_DIR) / source_folder

        # Đảm bảo thư mục đích tồn tại
        target_sub_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nKiểm tra thư mục: {target_sub_dir}")

        if not files_to_download:
            # Nếu danh sách tệp trống, tải toàn bộ thư mục con
            print(f"-> Tải toàn bộ thư mục '{source_folder}'...")
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=f"{source_folder}/*",
                local_dir=ROOT_DIR,
                local_dir_use_symlinks=False
            )
        else:
            # Tải các tệp cụ thể
            print(f"-> Tải các tệp được chỉ định trong '{source_folder or 'thư mục gốc'}'...")
            for one_file in files_to_download:
                file_path = target_sub_dir / one_file
                if file_path.exists():
                    print(f"  - Đã tồn tại: {one_file}")
                else:
                    download_with_progress(repo_id, source_folder, one_file, ROOT_DIR)


def download_hunyuan_avatar_model():
    """
    Tải xuống mô hình HunyuanVideo Avatar và tất cả các tệp phụ thuộc.
    """
    print("Bắt đầu quá trình tải xuống mô hình HunyuanVideo Avatar...")

    # === 1. Tải các tệp phụ thuộc chung được chia sẻ ===
    # Các tệp này cần thiết cho nhiều mô hình trong WanGP, bao gồm cả Hunyuan
    print("\n--- Giai đoạn 1: Tải các tệp phụ thuộc chung ---")
    shared_def = {
        "repo_id": "DeepBeepMeep/Wan2.1",
        "source_folder_list": ["pose", "depth", "mask", "wav2vec", ""],
        "file_list": [
            [],  # Tải toàn bộ thư mục 'pose'
            [],  # Tải toàn bộ thư mục 'depth'
            ["sam_vit_h_4b8939_fp16.safetensors"],
            # Các tệp cho Wav2Vec (dùng cho âm thanh)
            ["config.json", "feature_extractor_config.json", "model.safetensors", "preprocessor_config.json",
             "special_tokens_map.json", "tokenizer_config.json", "vocab.json"],
            ["flownet.pkl"]  # Dùng cho RIFE (upsampling)
        ]
    }
    process_files_def(**shared_def)

    # === 2. Tải các tệp cụ thể cho mô hình Hunyuan ===
    print("\n--- Giai đoạn 2: Tải các tệp của Hunyuan Video ---")

    # Chọn phiên bản mô hình. 'bf16' cho chất lượng cao hơn, 'quanto_bf16_int8' cho VRAM thấp hơn.
    # Chúng ta sẽ tải cả hai để bạn có thể chọn lúc chạy.
    transformer_filenames = [
        "ckpts/hunyuan_video_avatar_720_bf16.safetensors",
        "ckpts/hunyuan_video_avatar_720_quanto_bf16_int8.safetensors"
    ]

    # Hunyuan sử dụng LLaVA làm bộ mã hóa văn bản (text encoder)
    text_encoder_filename = "ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_fp16.safetensors"
    text_encoder_int8_filename = "ckpts/llava-llama-3-8b/llava-llama-3-8b-v1_1_vlm_quanto_int8.safetensors"

    hunyuan_def = {
        "repo_id": "DeepBeepMeep/HunyuanVideo",
        "source_folder_list": ["llava-llama-3-8b", "clip_vit_large_patch14", "whisper-tiny", "det_align", ""],
        "file_list": [
            # 1. Text Encoder (LLaVA-Llama-3-8B)
            ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
             "preprocessor_config.json"]
            + compute_list_from_path(text_encoder_filename)
            + compute_list_from_path(text_encoder_int8_filename),
            # 2. Vision Encoder (CLIP-ViT-Large-Patch14)
            ["config.json", "merges.txt", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json",
             "tokenizer.json", "tokenizer_config.json", "vocab.json"],
            # 3. Whisper (Dùng cho xử lý audio)
            ["config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json",
             "tokenizer_config.json"],
            # 4. Face Detection
            ["detface.pt"],
            # 5. Các tệp chính của Hunyuan (VAE, config, và mô hình transformer)
            [
                "hunyuan_video_720_quanto_int8_map.json",
                "hunyuan_video_custom_VAE_fp32.safetensors",
                "hunyuan_video_custom_VAE_config.json",
                "hunyuan_video_VAE_fp32.safetensors",
                "hunyuan_video_VAE_config.json"
            ] + compute_list_from_path(transformer_filenames[0]) + compute_list_from_path(transformer_filenames[1])
        ]
    }
    process_files_def(**hunyuan_def)

    print("\n=======================================================")
    print("Tất cả các tệp cho HunyuanVideo Avatar đã được tải xuống thành công!")
    print(f"Tất cả các mô hình đã được lưu trong thư mục: '{ROOT_DIR}'")
    print("=======================================================")


if __name__ == "__main__":
    download_hunyuan_avatar_model()