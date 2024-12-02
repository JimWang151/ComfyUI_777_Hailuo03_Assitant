# Made by Jim.Wang V1 for ComfyUI
import os
import subprocess
import importlib.util
import sys
import filecmp
import shutil

import __main__

python = sys.executable




from .ImageBridge import Hailuo03,ImageStitcher,PromptRefine

NODE_CLASS_MAPPINGS = {
    "GetNSFWPrompt": Hailuo03,
}


print('\033[34mHailuo03 Assistant Nodes: \033[92mLoaded\033[0m')