# runpod_serverless.py

import runpod
from handler import handler

runpod.serverless.start({"handler": handler})
