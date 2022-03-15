import json
import os
import pandas as pd
from sklearn.model_selection import KFold


# k_folder
def get_k_fold(csv_dir, k=5, shuffle=True, out_path='../data'):
    assert os.path.exists(csv_dir)
    assert k >= 2
    os.makedirs(out_path, exist_ok=True)

    df = pd.read_csv(csv_dir)
    kfold = KFold(k, shuffle=shuffle, random_state=1)
    for index, (train, val) in enumerate(kfold.split(df)):
        train_data = df.loc[train]
        val_data = df.loc[val]
        train_csv = pd.DataFrame(train_data)
        val_csv = pd.DataFrame(val_data)
        train_csv.to_csv(os.path.join(out_path, "train_split_{}.csv").format(index + 1))
        val_csv.to_csv(os.path.join(out_path, "val_split_{}.csv").format(index + 1))
    print("k fold down")


def get_classes_id_json(csv_file, save_dir):
    assert os.path.exists(csv_file)
    os.makedirs(save_dir, exist_ok=True)

    class_dict = {}
    df = pd.read_csv(csv_file)
    labels = df.groupby('individual_id').ngroup()
    for index, label in enumerate(labels):
        id = df['individual_id'][index]
        class_dict[id] = label
    # 生成类别名称以及对应的数字索引
    json_str = json.dumps(dict((key, val) for key, val in class_dict.items()), indent=4)
    with open(save_dir + '/class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    return class_dict


def get_input_data(csv_dir, img_dir, mask_dir):
    assert os.path.exists(csv_dir)
    assert os.path.exists(img_dir)

    df = pd.read_csv(csv_dir)
    image_name = df["image_id"].tolist()
    image_id = df["individual_id"].tolist()
    # image_path = [os.path.join(img_dir, x) for x in image_name]
    return {"name": image_name, "id": image_id, "img_path": img_dir, "mask_path": mask_dir}


if __name__ == '__main__':
    # get_k_fold(r"E:\dataset\happy_whale_cropped\train.csv")
    # data = get_input_data("../data/train_split_1.csv", "./", "./")
    get_classes_id_json(r"E:\dataset\happy_whale_cropped\train.csv", "./")
    print(1)
