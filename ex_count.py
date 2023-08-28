import json
import matplotlib.pyplot as plt

# 从文件中读取数据
with open('your_data_file.json', 'r') as file:
    data = json.load(file)

# 使用字典来统计每个exer_id被多少个user_id涉及
count_dict = {}

for user_data in data:
    done_exer_ids = set()  # 用于确保一个user_id只计算一次
    for log in user_data["logs"]:
        exer_id = log["exer_id"]
        if exer_id not in done_exer_ids:
            count_dict[exer_id] = count_dict.get(exer_id, 0) + 1
            done_exer_ids.add(exer_id)

# 从count_dict获取数据
exer_ids = list(count_dict.keys())
user_counts = list(count_dict.values())

plt.bar(exer_ids, user_counts)
plt.xlabel('Exer ID')
plt.ylabel('Number of Users')
plt.title('Number of Users for each Exer ID')
plt.xticks(rotation=45)  # 可能需要旋转x轴标签以获得更好的可读性
plt.tight_layout()  # 确保所有标签都可见
plt.show()
