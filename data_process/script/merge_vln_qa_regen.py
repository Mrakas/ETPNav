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

def get_qa_by_traj_id(qa_data, traj_id):
    for cur_data in qa_data:
        traj_id_qa = cur_data['image_name'].split('_')[0]
        traj_id_qa = int(traj_id_qa[1:])
        if traj_id == traj_id_qa:
            question = cur_data['question']
    return question

def get_gpt_res(prompt):
    """
    Sends a prompt to the GPT API and returns the generated response.

    :param prompt: The input prompt to send to the GPT model.
    :return: The generated response text.
    """
    API_KEY = "sk-IxyZ12cYdsxsvUCnD31eC59aFc1546Df8378302237125401"  # Replace with your API key
    BASE_URL = "https://api3.apifans.com/v1"  # Replace with your base URL
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    try:
        # Send the prompt to the GPT model
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo",  # Replace with the desired model
        )
        
        # Normalize the response text
        message = completion.choices[0].message
        result = unicodedata.normalize('NFKC', message.content)
        return result.strip()
    except Exception as e:
        print(f"Error during GPT API call: {e}")
        time.sleep(2)  # Retry delay
        return "Error: Unable to process the request."



PROMPT = """
I currently have a section of navigation instructions and a section with questions obtained based on visual information near the destination. Please help me combine the two sections and make the language more natural.
navigation instruction:"{instruction}"
question:"{question}"
answer in format :"merged data"
"""

step = 0

split = 'val_seen'


qa_file = f'/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/struct_from_gpt/output_step1_{split}.json'

#fix step
vln_file = f'/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/step2_gpt_merged/{split}_merged_fix20.json'
output_file_template = f'/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/step2_gpt_merged/{split}_merged_fix20_regen.json'

with open(qa_file, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)
with open(vln_file, 'r', encoding='utf-8') as f:
    vln_data = json.load(f)

for cur in tqdm(vln_data['episodes'][:]):
    step += 1
    ori_instruction_text = cur['instruction']['instruction_text']
    if "**Na" not in ori_instruction_text:
        continue
    
    traj_id = cur['trajectory_id']
    ori_qa = get_qa_by_traj_id(qa_data, traj_id)
    
    full_prompt = PROMPT.format(instruction=ori_instruction_text, question=ori_qa)
    #如果有 ** 那就重新生成.
    
    gpt_res = get_gpt_res(full_prompt)
    print("=====================")
    print("=====from=====", ori_instruction_text)
    print("=====to=====", gpt_res)
    # save in vln_data
    cur['instruction']['instruction_text'] = gpt_res
    print(f"step {step} done")
    #import ipdb; ipdb.set_trace()

    if True:

        with open(output_file_template, 'w', encoding='utf-8') as f:
            json.dump(vln_data, f, ensure_ascii=False, indent=4)
        print(f"Saved {step} steps to {output_file_template}")




print("finished")