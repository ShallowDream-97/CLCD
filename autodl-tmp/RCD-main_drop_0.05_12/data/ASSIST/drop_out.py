import json
import random

def load_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def random_delete_logs_from_data(data, proportion, seed=None):
    """
    Randomly delete a proportion of logs from the logs list for each user.
    """
    if not 0 <= proportion <= 1:
        raise ValueError("Proportion should be between 0 and 1")

    random.seed(seed)

    for user_data in data:
        logs = user_data['logs']
        n_delete = int(len(logs) * proportion)
        indices_to_delete = random.sample(range(len(logs)), n_delete)
        
        # Update logs by excluding deleted records
        user_data['logs'] = [item for index, item in enumerate(logs) if index not in indices_to_delete]
        user_data['log_num'] = len(user_data['logs'])

    return data

# 使用示例

# 从json文件加载数据
data = load_from_json("log_data.json")

# 在每个用户的logs中随机删除50%的记录，并设置随机种子为42
reduced_data = random_delete_logs_from_data(data, 0.05, seed=12)

# 将处理后的数据保存到新的json文件
save_to_json(reduced_data, "0.05_12_log_data.json")
