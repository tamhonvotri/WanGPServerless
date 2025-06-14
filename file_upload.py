import uuid, requests, pathlib

ACCESS_KEY = "17e23633-2a7a-4d29-9450be4d6c8e-e01f-45f4"
LOCAL_FILE = pathlib.Path("myvideo.mp4")
unique_key = uuid.uuid4().hex                # có thể dùng tên khác
url = f"https://storage.bunnycdn.com/zockto/video/{unique_key}.mp4"

with LOCAL_FILE.open("rb") as f:
    r = requests.put(
        url,
        headers={
            "AccessKey": ACCESS_KEY,
            "Content-Type": "video/mp4"
        },
        data=f,               # stream theo chunk, không load hết vào RAM
        timeout=1200          # tăng timeout nếu file lớn
    )
print(r.status_code, r.text)
print("CDN URL:", f"https://zockto.b-cdn.net/video/{unique_key}.mp4")
