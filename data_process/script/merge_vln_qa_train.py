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
    API_KEY = "sk-KlQFn0xmaR52YmJNwH4hXjLbK6m4kPJo8J8JXtjAI213qgr0"  # Replace with your API key
    BASE_URL = "https://lonlie.plus7.plus/v1"  # Replace with your base URL
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
            model="gpt-3.5-turbo-1106",  # Replace with the desired model
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
"""

step = 0

split = 'train'
output_file_template = f'/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/step2_gpt_merged/{split}_merged.json'
qa_file = f'/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/struct_from_gpt/output_step1_{split}.json'
vln_file = f'/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/R2R_VLNCE_v1-2_preprocessed/{split}/{split}.json'
with open(qa_file, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)
with open(vln_file, 'r', encoding='utf-8') as f:
    vln_data = json.load(f)

for cur in tqdm(vln_data['episodes']):
    step += 1
    ori_instruction_text = cur['instruction']['instruction_text']
    traj_id = cur['trajectory_id']
    ori_qa = get_qa_by_traj_id(qa_data, traj_id)
    
    full_prompt = PROMPT.format(instruction=ori_instruction_text, question=ori_qa)
    gpt_res = get_gpt_res(full_prompt)

    # save in vln_data
    cur['instruction']['instruction_text'] = gpt_res
    print(f"step {step} done")
    #import ipdb; ipdb.set_trace()
    if step % 20 == 0:
        part_number = step // 20
        output_file = output_file_template.format(part_number)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vln_data, f, ensure_ascii=False, indent=4)
        print(f"Saved {step} steps to {output_file}")




print("finished")