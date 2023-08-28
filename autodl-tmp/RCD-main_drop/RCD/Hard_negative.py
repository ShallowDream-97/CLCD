import copy
import random
import json

# 读取现有的数据
with open('../data/ASSIST/log_data.json', 'r') as file:
    data = json.load(file)

# 设定参数
student_selection_ratio = 0.2  # 要选择的学生的比例
log_selection_ratio = 0.5      # 每个学生中要修改的log的比例

# 获取现有的学生数量
num_students = len(data)

# 随机选择要复制和修改的学生
selected_indices = random.sample(range(num_students), int(num_students * student_selection_ratio))
new_data = data
# 复制和修改选定的学生
for index in selected_indices:
    # 创建学生的深拷贝，以便修改
    new_student = copy.deepcopy(data[index])
    
    # 生成新的用户ID
    new_student["user_id"] = len(new_data) + 1

    # 随机选择要修改的log
    selected_logs = random.sample(new_student["logs"], int(len(new_student["logs"]) * log_selection_ratio))

    # 修改选定的log
    for log in selected_logs:
        log["score"] = 1 - log["score"]

    # 将新的学生添加到现有数据中
    new_data.append(new_student)

# 现在，数据中包含了原始的学生和新的学生
# 你可以将其保存为新的JSON文件
import json

with open("new_data.json", "w") as file:
    json.dump(new_data, file, indent=4)
