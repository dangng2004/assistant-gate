# import hydra
# from omegaconf import DictConfig
# import argparse
# import fire
# import pandas as pd
# from tqdm import tqdm
# import numpy as np
# import json
# import logging
# import torch
# import random
# import re
# import time
# import os
# import signal
# from collections import defaultdict
# from datasets import load_dataset, Dataset
# import requests

# from utils import *

# # import models
# from AG.models.huggingface.hf_inference_model import HFInferenceModel
# from AG.models.openai.azure import AsyncAzureChatLLM
# from AG.models.openai.gpt4 import GPT4Agent
# from AG.models.vllm_models.inference_model import VLLMInferenceModel
# from paths import *

# # logging
# logging.basicConfig(level=logging.INFO)

# @hydra.main(version_base=None, config_path="conf", config_name='train-split-config')
# def main(args: DictConfig) -> None:
#     logging.info(f"Loading GPT-4 and generating personas for {args.split.name} split...")
#     random.seed(1)
    
#     # GPT-4 support only atm
#     llm = AsyncAzureChatLLM(**args.model.model_config.azure_api)
#     model = GPT4Agent(llm=llm, **args.model.run.completion_config)

#     # Load expert personas from PRODIGy dataset
#     # URL to the raw JSON file
#     url = 'https://raw.githubusercontent.com/LanD-FBK/prodigy-dataset/main/dataset/characters.json'

#     # Send a GET request to the URL
#     response = requests.get(url)

#     # Check if the request was successful
#     if response.status_code == 200:
#         # Parse the JSON content
#         data = response.text.strip()
#         data = '{' + data[1:len(data) - 1] + '}'
#         data = json.loads(data)
        
#     else:
#         print("Failed to retrieve the JSON file.")
        
#     interest_keys = ['u0', 'u9', 'u498', 'u838', 'u861', 'u1106', 'u1172', 'u1273', 'u1323', 'u1456', 'u1767', 'u2563', 'u2585', 'u2746', 'u3105', 'u3125', 'u3171', 'u3283', 'u3321', 'u3667', 'u3681']
#     # Filter data by keys based on the keys above
#     data = dict([(k, v) for k, v in data.items() if k in interest_keys])
#     # Preprocess
#     expert_personas = [f"I'm {char_data['character_name']}. {' '.join(char_data['biography'])}" for u, char_data in data.items()]    
    
#     # train split
#     messages = []
#     for i in range(args.SHOT_GROUPS):
#         message = GENERATION_PROMPTS[args.PROMPT_IDX] + '\n\n'
#         few_shot_indices = random.sample(range(0, len(expert_personas)), 2)
#         few_shots = '\n\n'.join([expert_personas[j] for j in few_shot_indices])
#         message += few_shots + '\n\n'
#         messages.append(message)

#     # completions is a list containing len(messages) lists, each of size args.model.run.completion_config.n
#     completions = model.batch_prompt(
#                     system_message=SYS_MSGS[args.SYS_IDX],
#                     messages=messages,
#                 )
#     flattened_completions = [item for sublist in completions for item in sublist]
#     if not os.path.exists(f'{PERSONAS_PATH}/{VERSION_2_BSFT}'):
#         os.makedirs(f'{PERSONAS_PATH}/{VERSION_2_BSFT}')
    
#     with open(f'{PERSONAS_PATH}/{VERSION_2_BSFT}/{args.split.name}.json', 'w') as f:
#         json.dump(flattened_completions, f)    

# if __name__ == '__main__':
#     try:
#         fire.Fire(main())
#     except:
#         pass

import hydra
from omegaconf import DictConfig
import logging
import random
import os
import json
import requests

from openai import OpenAI
from utils import *
from paths import *

# Configure logging
logging.basicConfig(level=logging.INFO)
# Initialize OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI()

@hydra.main(version_base=None, config_path="conf", config_name='train-split-config')
def main(args: DictConfig) -> None:
    logging.info(f"Loading GPT-4 and generating personas for {args.split.name} split...")
    random.seed(1)

    # Extract completion settings
    model_name  = args.model.run.completion_config.model
    temperature = args.model.run.completion_config.temperature
    n           = args.model.run.completion_config.n
    max_tokens  = getattr(args.model.run.completion_config, 'max_tokens', 512)

    # Fetch expert personas JSON
    url = 'https://raw.githubusercontent.com/LanD-FBK/prodigy-dataset/main/dataset/characters.json'
    resp = requests.get(url)
    resp.raise_for_status()
    raw = resp.text.strip()
    raw = '{' + raw[1:-1] + '}'
    data = json.loads(raw)

    # Filter for keys of interest
    interest_keys = [
        'u0','u9','u498','u838','u861','u1106','u1172','u1273',
        'u1323','u1456','u1767','u2563','u2585','u2746','u3105',
        'u3125','u3171','u3283','u3321','u3667','u3681'
    ]
    filtered = {k: v for k, v in data.items() if k in interest_keys}
    expert_personas = [
        f"I'm {v['character_name']}. {' '.join(v['biography'])}"
        for v in filtered.values()
    ]

    # Build few-shot prompts
    messages = []
    for _ in range(args.SHOT_GROUPS):
        prompt = GENERATION_PROMPTS[args.PROMPT_IDX]
        idxs = random.sample(range(len(expert_personas)), 2)
        shots = "\n\n".join(expert_personas[i] for i in idxs)
        messages.append(f"{prompt}\n\n{shots}\n\n")

    # Call OpenAI SDK for chat completions
    all_completions = []
    for user_msg in messages[:5]:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYS_MSGS[args.SYS_IDX]},
                {"role": "user",   "content": user_msg}
            ],
            temperature=temperature,
            n=n,
            max_tokens=max_tokens
        )
        texts = [choice.message.content for choice in response.choices]
        all_completions.append(texts)

    # Flatten and write to disk
    flattened = [text for sub in all_completions for text in sub]
    output_dir = os.path.join(PERSONAS_PATH, VERSION_2_BSFT)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{args.split.name}.json")
    with open(out_path, 'w') as f:
        json.dump(flattened, f, indent=2)

if __name__ == '__main__':
    main()



