import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_id_list(mask_path, class_list):
    assert os.path.exists(mask_path)
    if isinstance(class_list, int):
        class_list = [class_list]

    id_list = []
    for file in tqdm(os.listdir(mask_path)):
        path = os.path.join(mask_path, file)
        mask = Image.fromarray(np.array(Image.open(path)))
        classes = set(np.unique(mask)) & set(class_list)
        if not classes:
            continue
        id_list.append(file.split('.')[0])
    return id_list


def main(args):
    trn = os.path.join(args.out_path, "trn")
    val = os.path.join(args.out_path, "val")
    test = os.path.join(args.out_path, "test")
    os.makedirs(trn, exist_ok=True)
    os.makedirs(val, exist_ok=True)
    os.makedirs(test, exist_ok=True)

    assert os.path.exists(args.anno_path), f"{args.anno_path} dose not exists!"
    # anno_path = r'E:\practice\DeepLearning\Image_segmentation\few_shot_segmentation\SegmentationClass'

    for fold in range(args.fold):
        nclass_trn = args.nclass // args.fold
        class_ids_val = [fold * nclass_trn + i + 1 for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(1, args.nclass + 1) if x not in class_ids_val]

        trn_f = open(os.path.join(trn, f"fold{fold}.txt"), 'w')
        val_f = open(os.path.join(val, f"fold{fold}.txt"), 'w')

        trn_id_list = load_id_list(mask_path=args.anno_path, class_list=class_ids_trn)
        val_id_list = load_id_list(mask_path=args.anno_path, class_list=class_ids_val)
        print("\nfold:{} train class: {} val class: {} train data len: {} val data len: {}".format(fold, class_ids_trn,
                                                                                                   class_ids_val,
                                                                                                   len(trn_id_list),
                                                                                                   len(val_id_list)))

        for i in trn_id_list:
            trn_f.write(i.strip() + "\n")
        for i in val_id_list:
            val_f.write(i.strip() + "\n")

        trn_f.close()
        val_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Split data")
    parser.add_argument('--anno_path', type=str, help='path of mask data', required=True)
    parser.add_argument('--out-path', type=str, help='output path', default='output')
    parser.add_argument('--fold', type=int, default=4)
    parser.add_argument('--nclass', type=int, required=True)

    args = parser.parse_args()
    main(args)
