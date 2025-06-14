################################
#          handler.py          #
################################
"""
Handler RunPod Serverless:
- Nhận input JSON chứa image_url, audio_url, text
- Tải file về /tmp
- Gọi run_avatar_cli.py -> trả stdout (hoặc URL video nếu script tự upload)
"""

import os
import re
import sys
import json
import subprocess
import requests
from pathlib import Path

TMP_DIR = "/tmp"
SCRIPT = Path("python run_avatar_cli.py")  # đường dẫn cố định

subprocess.run(
    ["python", "download_model_cli.py"],
    check=True
)

def _download(url: str, fname: str) -> str:
    """Tải url về /tmp rồi trả lại path."""
    dst = Path(TMP_DIR) / fname
    with requests.get(url, timeout=60, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    return str(dst)


def handler(event):
    try:
        inp = event.get("input", {})
        prompt = inp.get("prompt", "")

        # 1. Tải file từ URL (nếu có)
        img_path = _download(inp["image_url"], "input.png") if "image_url" in inp else None
        aud_path = _download(inp["audio_url"], "input.wav") if "audio_url" in inp else None


        cmd = [
            "python",  # Python interpreter hiện tại
            "run_avatar_cli.py",  # File CLI cần chạy
            "--image", "input.png",
            "--audio", "input.wav",
            "--prompt", prompt,
        ]
        print(cmd)
        result = subprocess.run(
            cmd,
            capture_output=True,  # lấy cả stdout và stderr
            text=True,  # trả về str thay vì bytes
            check=True  # tự động raise CalledProcessError nếu lỗi
        )
        # return result.stdout

        # 3. Thử parse stdout thành JSON, không được thì trả raw string
        try:
            result = json.loads(result.stdout)
        except json.JSONDecodeError:
            result = result.stdout.strip()
        # match = re.search(r"CDN URL:\s*(https?://[^\s]+)", result.stdout)
        # if match:
        #     return {"cdn_url": match.group(1)}
        # else:
        #     return {"error": "CDN URL not found", "raw_output": result.stdout}
        return {"result": result}

    except subprocess.CalledProcessError as e:
        return {
            "error": "run_avatar_cli.py exited with non-zero code",
            "code":  e.returncode,
            "stdout": e.stdout,      # thêm dòng này
            "stderr": e.stderr,      # và giữ stderr
        }
    except Exception as e:
        return {"error": str(e)}
