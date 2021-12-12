import os
import argparse
from tqdm import tqdm
import glob


def main(args):
    # 这里使用的是花数据集，按需求重写该函数
    assert os.path.exists(args.data_path), f"error data path: {args.data_path} does not available"
    train_dir = args.data_path + "/train"
    test_dir = args.data_path + "/test"
    train_txt = open(os.path.join(args.txt_out_path, "train.txt"), 'w')
    val_txt = open(os.path.join(args.txt_out_path, "val.txt"), 'w')
    for path in tqdm(glob.glob(train_dir + '/*/*')):
        train_txt.write(path.strip()+'\n')
    for path in tqdm(glob.glob(test_dir+'/*')):
        val_txt.write(path.strip()+'\n')
    train_txt.close()
    val_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default=r'D:\dataset\flow_data')
    parser.add_argument('--txt-out-path', default='./')

    args = parser.parse_args()
    main(args)
