import json
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

def extract_unique_words(sentence_list, lowercase=True):
    """提取并返回唯一单词集合"""
    word_set = set()
    for sentence in sentence_list:
        # 使用NLTK进行可靠的分词（保留连字符单词）
        tokens = word_tokenize(sentence)
        # 过滤纯标点符号
        words = [w for w in tokens if w.isalnum() or '-' in w]
        # 可选：统一小写
        if lowercase:
            words = [w.lower() for w in words]
        word_set.update(words)
    return sorted(word_set)  # 排序保证可复现性

# 使用示例




# 文件路径
paths = [
    "/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/step3_clean_merged/train_merged_fullyclean.json",
    "/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/step3_clean_merged/val_seen_merged_fullyclean.json",
    "/mnt/data5/ghx/ETPworkplace/ETPNav/data_process/step3_clean_merged/val_unseen_merged_fullyclean.json"
]



texts = []
print("Started == Loaded weights")

# 读取数据并提取文本
for path in paths:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for episode in data['episodes']:
            text = episode['instruction']['instruction_text']
            texts.append(text)
            # if 'chio' in text:
            #     import ipdb; ipdb.set_trace()

# 原始词汇表
ori_word_list = data['instruction_vocab']['word_list']
import ipdb; ipdb.set_trace()
never_split_words = extract_unique_words(texts) + ori_word_list
# 合并原始词汇表和提取的文本
texts = ori_word_list + texts
import ipdb; ipdb.set_trace()
# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained(
    'google-bert/bert-base-uncased', do_basic_tokenize=True, never_split=never_split_words)
#import ipdb; ipdb.set_trace()
print("Started tokenization")

# 对文本进行分词
tokenized_texts = [tokenizer.tokenize(text) for text in texts]

# 展平分词结果到一个列表中
all_tokens = [token for tokens in tokenized_texts for token in tokens]

# 构建唯一单词列表
word_list = sorted(set(all_tokens))  # 按字母顺序排序，方便调试和复现

# 构建 word2idx 字典
word2idx_dict = {word: idx for idx, word in enumerate(word_list)}

# 保存结果为 JSON 文件
output = {
    "word_list": word_list,
    "word2idx_dict": word2idx_dict
}
#import ipdb; ipdb.set_trace()
with open("word_list_and_word2idx.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
print("len:", len(word_list))
print("Finished! Results saved in 'word_list_and_word2idx.json'")