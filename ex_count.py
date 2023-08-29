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

# 对exer_ids进行排序
sorted_exer_ids = sorted(list(count_dict.keys()))

# 按每10个exer_id合并为一个区间，并加总用户数
merged_exer_intervals = []
merged_user_counts = []
for i in range(0, len(sorted_exer_ids), 10):
    start = sorted_exer_ids[i]
    end = sorted_exer_ids[min(i + 9, len(sorted_exer_ids) - 1)]
    merged_exer_intervals.append(f"{start}-{end}")
    merged_user_counts.append(sum([count_dict[exer_id] for exer_id in sorted_exer_ids[i:i+10]]))

plt.bar(merged_exer_intervals, merged_user_counts)
plt.xlabel('Exer ID Interval')
plt.ylabel('Number of Users')
plt.title('Number of Users for each Exer ID Interval')
plt.xticks(rotation=45)
plt.tight_layout()

# 保存图像
plt.savefig('output_merged_filename.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
