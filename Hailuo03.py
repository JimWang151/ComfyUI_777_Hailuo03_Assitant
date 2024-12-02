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
                    "skincolor" : ("STRING", {"multiline": True,  "tooltip": "The text to be encoded."}),
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

        def refine(self, step, character,scenes,skincolor,width,height):
            # 初始化画布高度和宽度

            prompt="raw photo,";

            prompt+=self.getShotStyle(step,width,height)
            # 返回结果
            prompt+=self.getAperture(step,width,height)
            prompt+=character+" with seductive  expression,"+self.getPoseAndFacing(step,width,height,skincolor)
            prompt += scenes
            prompt += self.getFocalLength(step,width, height)
            prompt += "with "+skincolor+" skin,"
            prompt+=self.getLightCondition(step,width,height,scenes)
            #print(f"final prompt: {prompt}")
            return (prompt,)

        def getShotStyle(self,step,width,height):
            # 获取镜头类型
            result = ""  # 初始化结果字符串为空
            if height==1024 :
                result="full body shot,perspective Compression,"
            # 检查是否满足特定的条件，并追加相应的字符串
            if width == 1024 and step % 2 == 0:  # 检查step是否可以被2整除
                result += "side view,"
            if width == 768 and height == 512:
                result = "(half body:1.3),"
                if step % 2 ==0:
                    result="Close-up of her ass and clearly pussy ,"
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
                    result ="f/3-f/6,"
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

        def getPoseAndFacing(self,step, width,height,atmosphere):
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