from datasets import load_dataset

# 指定存储路径
save_path = "./data/hotpotqa"

# 加载并下载到指定位置
# dataset = load_dataset(
#     "hotpotqa/hotpot_qa", 
#     "distractor", 
#     cache_dir=save_path
# )

# dataset = load_dataset(
#     "hotpotqa/hotpot_qa", 
#     "fullwiki", 
#     cache_dir=save_path
# )

wiki_corpus = load_dataset(
    "TIGER-Lab/LongRAG", 
    "hotpot_qa_wiki", 
    cache_dir=save_path
)

print(f"数据集已下载并缓存至: {save_path}")