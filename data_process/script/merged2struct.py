import json
import os
import time
from openai import OpenAI
import unicodedata
from tqdm import tqdm
import re
"""
merge vln and qa json files with gpt, save in vln file. 

用于正则化提取
先提取 "xxx"
然后提取 ---xxx---
删除文本中的 \n 
"""

def clean_instruction(text):
    if '"' in text:
        result = re.search(r'\"(.*?)\"', text, re.DOTALL)
        print(result)
        if result:
            return result.group(1)
        else:
            print("error")
            import ipdb; ipdb.set_trace()

    elif '---' in text:
        result = re.search(r'---(.*?)---', text, re.DOTALL)
        if result:
            return result.group(1)
        else:
            result = re.search(r'---(.*?)', text, re.DOTALL)
            if result:
                return result.group(1)
            else:
                print("error")
                import ipdb; ipdb.set_trace()

    else:
        return text
    

step = 0

split = 'train'
vln_file =    f'/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/step2_gpt_merged/{split}_merged_fix20_regen.json'
output_file = f'/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/step2_gpt_merged/{split}_merged_fullyclean.json'



with open(vln_file, 'r', encoding='utf-8') as f:
    vln_data = json.load(f)

for cur in tqdm(vln_data['episodes'][:]):
    step += 1
    ori_instruction_text = cur['instruction']['instruction_text']

    # save in vln_data
    cur['instruction']['instruction_text'] = clean_instruction(ori_instruction_text)
    
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(vln_data, f, ensure_ascii=False, indent=4)


print("finished, saved to", output_file)