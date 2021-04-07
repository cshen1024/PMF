import numpy as np
import pandas as pd

def data_process(data_path):
    raw_data = pd.read_csv(data_path, sep='\t',
                           names=['user_id', 'item_id', 'rating', 'timestamp'])

    unique_users = list(set(raw_data['user_id']))
    unique_items = list(set(raw_data['item_id']))
    dict_userid_to_index = dict((userid, index) for userid, index in zip(unique_users, range(len(unique_users))))
    dict_itemid_to_index = dict((itemid, index) for itemid, index in zip(unique_items, range(len(unique_items))))

    data = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip().split('\t')
            user_id = int(line[0])
            item_id = int(line[1])
            rating = float(line[2])

            data.append([dict_userid_to_index[user_id], dict_itemid_to_index[item_id], rating])

    data = np.array(data)
    np.random.shuffle(data)

    return dict_userid_to_index, dict_itemid_to_index, data

def RMSE(preds, truth):
    return np.sqrt(np.mean(np.square(preds-truth)))