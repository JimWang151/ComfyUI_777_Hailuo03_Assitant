# Made by JimWang for ComfyUI
# 02/04/2023

import torch
import random
import folder_paths
import uuid
import json
import urllib.request
import urllib.parse
import os
import numpy as np

import comfy.utils
from comfy.cli_args import args

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import collections
from torchvision.transforms import ToPILImage,ToTensor
import torchvision.transforms as T

from PIL import Image, ImageDraw

import collections


class GetNSFWPrompt:
        def __init__(self):
            pass

        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "character": ("STRING", {"multiline": True,  "tooltip": "The text to be encoded."}),
                    "scenes" : ("STRING", {"multiline": True,  "tooltip": "The text to be encoded."}),
                    "width": ("INT",{"default": 512}),
                    "height": ("INT",{"default": 512}),
                    "step": ("INT", {"default": 1}),
                },
            }

        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("prompt",)
        FUNCTION = "refine"
        OUTPUT_NODE = True
        CATEGORY = "Hailuo01"
        DESCRIPTION = "Generate prompt."

        def refine(self, step, character,scenes,width,height):
            # 初始化画布高度和宽度

            prompt="raw photo,";

            prompt+=self.getShotStyle(step,width,height)
            # 返回结果
            if width == 768 and height == 512 and step % 2 == 0:
                prompt+=self.insert_after_woman_or_girl(character)
            else:
            #prompt+=self.getAperture(step,width,height)
                prompt+=character+" with seductive  expression,"+self.getPoseAndFacing(step,width,height)
            prompt += scenes
           # prompt += self.getFocalLength(step,width, height)
            prompt+=self.getLightCondition(step,width,height,scenes)

            #print(f"final prompt: {prompt}")
            return (prompt,)

        def getShotStyle(self,step,width,height):
            # 获取镜头类型
            result = ""  # 初始化结果字符串为空
            if height==1024 and width==768:
                result="full body shot,"
            # 检查是否满足特定的条件，并追加相应的字符串
            if height == 1024 and width==1024 and step % 2 == 0:  # 检查step是否可以被2整除
                result += "full body with visible pussy,"
            if width == 768 and height == 512:
                result = "(half body with breast:1.3),"
                if step % 2 ==0:
                    result="close-up of "
            # 返回结果
            return result


        def getAperture(self,step, width,height,round=1):
            # 获取光圈
            result=""
            if height == 1024:
                result="f/10,"
            if height == 1024 and width ==1024 :
                result = "f/5.6-f8,"
            if width == 768 and height == 512 :
                result = "f/11,"
                if step % 2 ==0:
                    result =""
            # 返回结果
            return result

        def getFocalLength(self,step, width,height,round=1):
            # 获取焦距
            result = ""
            if height == 1024:
                result=",50mm,"
                if height == 1024 and width == 1024:
                    result = ",85mm,"
            if width == 768 and height == 512 :
                result = ",85mm,"
                if step % 2 ==0:
                    result =",200mm,"
            # 返回结果

            return result

        def getPoseAndFacing(self,step, width,height):
            # 聚焦肯虚化
            result=""
            if height == 1024:
                result=",natural waist pose,natural leg pose,natural face side,"
            if width == 768 and height == 512 :
                result = ",looking somewhere,"
                if step % 2 ==0:
                    result =","
            # 返回结果
            return result
        def getLightCondition(self,step, width,height,scenes):
            # 聚焦肯虚化
            result=",enough lighting,outdoor natural light,"

            # 返回结果
            return result

        def insert_after_woman_or_girl(self,text):
            # 检查字符串长度是否足够
            if len(text) < 12:
                return text

            # 截取前12个字符
            first_12_chars = text[:18]

            # 检查是否包含 "woman" 或 "girl"
            if "woman" in first_12_chars:
                index = first_12_chars.index("woman")
                insert_text = "'s ass with clearly pussy,fucking ,"
            elif "girl" in first_12_chars:
                index = first_12_chars.index("girl")
                insert_text = "'s ass with clearly pussy,fucking ,"
            else:
                return text

            # 计算插入位置
            insert_index = index + len("woman") if "woman" in first_12_chars else index + len("girl")

            # 插入文本并返回结果
            return text[:insert_index] + insert_text + text[insert_index:]