import os
import copy
import glob
import random
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Image Response")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--images-dir", required=True, help="Path to the directory containing images.")
    parser.add_argument("--prompt", required=True, help="User prompt/question for LLM.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save LLM's responses.")
    parser.add_argument("--options", nargs="+", help="override some settings in the used config.")
    args = parser.parse_args()
    return args

def setup_seeds():
    seed = 1

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
setup_seeds()

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# # Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Get all images
images = []
extensions = ['*.png', '*.jpg', '*.JPG']
for ext in extensions:
    images.extend(glob.glob(f"{args.images_dir}/{ext}"))
images = sorted(images)

print(f"Found: {len(images)} images - {args.images_dir}/")

# Get the intial chat state
initial_chat_state = CONV_VISION.copy()

# For  each image
print("Processing each of the images")
for image_file in tqdm(images):

    # Load Image
    img = Image.open(image_file)
    img = Image.open(image_file).convert("RGB")

    # Create another state
    current_state = copy.deepcopy(initial_chat_state)

    # Upload the image
    img_list = []
    llm_message = chat.upload_img(img, current_state, img_list)

    # Get response
    chat.ask(args.prompt, current_state)
    llm_response = chat.answer(conv=current_state,
                                img_list=img_list,
                                num_beams=5,
                                temperature=1.0,
                                max_new_tokens=300,
                                max_length=2000)[0]
        
    # Save the response to a file
    output_file_path = os.path.join(args.output_dir,f"{os.path.basename(image_file)[:-4]}_output.txt")
    with open(output_file_path, 'w') as f:
        f.write(llm_response)
