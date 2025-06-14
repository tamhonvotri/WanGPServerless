FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Cài công cụ hệ thống, dev tool, CUDA env
#RUN add-apt-repository ppa:savoury1/ffmpeg4 -y
RUN  apt-get update && apt-get install 'ffmpeg' -y

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

#WORKDIR /workspace

#RUN git clone https://github.com/deepbeepmeep/Wan2GP.git
#COPY run_avatar_cli.py /workspace/Wan2GP/
#COPY download_model_cli.py /workspace/Wan2GP/
#COPY handler.py /workspace/Wan2GP/
#COPY runpod_serverless.py /workspace/Wan2GP/
WORKDIR /runpod-volume/Wan2GPCLI/Wan2GP
RUN pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124
#RUN pip install -r requirements.txt
RUN pip install --no-cache-dir \
  "torch>=2.4.0" \
  "torchvision>=0.19.0" \
  "opencv-python>=4.9.0.80" \
  "diffusers>=0.31.0" \
  "transformers==4.51.3" \
  "tokenizers>=0.20.3" \
  "accelerate>=1.1.1" \
  "tqdm" "imageio" "easydict" "ftfy" "dashscope" "imageio-ffmpeg" \
  "gradio==5.23.0" \
  "numpy>=1.23.5,<2" \
  "einops" "moviepy==1.0.3" "mmgp==3.4.8" "peft==0.14.0" \
  "mutagen" "pydantic==2.10.6" "decord" "onnxruntime-gpu" \
  "rembg[gpu]==2.0.65" "matplotlib" "timm" "segment-anything" \
  "omegaconf" "hydra-core" "librosa" "loguru" "sentencepiece" "av"
RUN pip install runpod


RUN pip install sageattention==1.0.6
#RUN python download_model_cli.py


#WORKDIR /runpod-volume/Wan2GPCLI


CMD ["python", "runpod_serverless.py"]
