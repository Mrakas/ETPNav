import json
import gzip
import math
import sys
sys.path.append('/mnt/data5/ghx/ETPworkplace/ETPNav')

import numpy as np
import math

def heading2quaternion(heading):
    half_angle = -heading/2
    w = math.cos(half_angle)
    x = 0.0  # 不绕x轴旋转
    y = math.sin(half_angle)  # 绕y轴旋转
    z = 0.0  # 不绕z轴旋转
    return [x, y, z, w]

split = 'train'
path = f"/mnt/data5/ghx/ETPworkplace/ETPNav/data_backup/preds_AEQA_{split}_s1.json"
path_goal = f'/mnt/data5/ghx/ETPworkplace/ETPNav/data/datasets/R2R_VLNCE_v1-2_preprocessed_AEQA/{split}/{split}_bertidx.json.gz'
out_path = f'/mnt/data5/ghx/ETPworkplace/ETPNav/data/datasets/model_goal_to_gt/{split}_s1.json.gz'


with open(path, "r") as f: #episode_id 是对应的
    data_dict = json.load(f)

# .json.gz
with gzip.open(path_goal, "r") as f:
    raw_data_goal = json.load(f)
    data_goal = raw_data_goal['episodes']

if len(data_dict) != len(data_goal):
    print(len(data_dict))
    print(len(data_goal))
    raise ValueError("data_dict and data_goal have different length")

for episode in data_goal:
    #import ipdb; ipdb.set_trace()

    episode_id = episode['episode_id']
    episode['start_position'] = data_dict[str(episode_id)][-1]['position'] # 最后一个点 当起点
    episode['start_rotation'] = heading2quaternion(data_dict[str(episode_id)][-1]['heading'])

#save sa gzip
with gzip.open(out_path, "wt") as f:
    json.dump(raw_data_goal, f)


print(f"save to {out_path}")


