import json

# 从文件中读取JSON数据
with open('../data/ASSIST/_train_data.json', 'r') as file:
    data = json.load(file)
    

# 查找最大的exer_id
max_exer_id = max(item['exer_id'] for item in data)

# 打印最大的exer_id
print("最大的user_id是:", max_exer_id)

# 使用集合存储不同的exer_id
unique_exer_ids = set()

# 遍历数据集中的每一项
for item in data:
    exer_id = item['user_id']
    unique_exer_ids.add(exer_id)

# 打印不同exer_id的数量
print("不同的user_id数量是:", len(unique_exer_ids))

# 使用集合存储不同的knowledge_code
unique_knowledge_codes = set()

# 遍历数据集中的每一项
for item in data:
    knowledge_codes = item['knowledge_code']
    for code in knowledge_codes:
        unique_knowledge_codes.add(code)

# 打印不同knowledge_code的数量
print("不同的knowledge_code数量是:", len(unique_knowledge_codes))