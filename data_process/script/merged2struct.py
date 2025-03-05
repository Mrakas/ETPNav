import json
import os
import time
import openai
from openai import OpenAI
import unicodedata
from tqdm import tqdm
"""
merge vln and qa json files with gpt, save in vln file. 
"""

def clean_instruction(instruction):
    pass
    instruction = instruction.replace('\n', ' ')
    instruction = instruction.replace('  ', ' ')
    instruction = instruction.strip()
    return instruction

step = 0

split = 'val_seen'
vln_file =    f'/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/step2_gpt_merged/{split}_merged_fix20.json'
output_file = f'/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/step2_gpt_merged/{split}_merged_clean.json'



with open(vln_file, 'r', encoding='utf-8') as f:
    vln_data = json.load(f)

for cur in tqdm(vln_data['episodes'][:]):
    step += 1
    ori_instruction_text = cur['instruction']['instruction_text']

    # save in vln_data
    #cur['instruction']['instruction_text'] = clean_instruction(ori_instruction_text)
    print("=====================")
    print(cur['instruction']['instruction_text'])
    #import ipdb; ipdb.set_trace()
    
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(vln_data, f, ensure_ascii=False, indent=4)


print("finished, saved to", output_file)