import os
from loguru import logger

IMG_FORMATS = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo']  # acceptable image suffixes


@logger.catch
class Dataset(object):
    def __init__(self, dataset_dir, mode):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.train_dir = os.path.join(dataset_dir, "train")
        self.query_dir = os.path.join(dataset_dir, "query")
        self.gallery_dir = os.path.join(dataset_dir, "gallery")
        self._check_before_run()
        train_relabel = (mode == 'retrieval')

        train, train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=train_relabel)
        query, query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)

        num_train_pids = len(train_pids)
        num_query_pids = len(query_pids)
        num_gallery_pids = len(gallery_pids)
        num_total_pids = len(set(train_pids) | set(query_pids) | set(gallery_pids))
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        logger.info("==> dataset loaded")
        logger.info("Dataset statistics:")
        logger.info("  ------------------------------")
        logger.info("  subset   | # ids | # images")
        logger.info("  ------------------------------")
        logger.info("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        logger.info("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        logger.info("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        logger.info("  ------------------------------")
        logger.info("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        logger.info("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        assert os.path.exists(self.dataset_dir), "ERROR {} dose not exists".format(self.dataset_dir)
        assert os.path.exists(self.train_dir), "ERROR {} dose not exists".format(self.train_dir)
        assert os.path.exists(self.query_dir), "ERROR {} dose not exists".format(self.query_dir)
        assert os.path.exists(self.gallery_dir), "ERROR {} dose not exists".format(self.gallery_dir)

    def _process_dir(self, dir_path, relabel=False):
        cats = os.listdir(dir_path)
        pid_container = set()
        dataset = []

        for cat in cats:
            pid = int(cat.strip().split("_")[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for cat in cats:
            pid = int(cat.strip().split("_")[0])
            if relabel:
                pid = pid2label[pid]
            next_path = os.path.join(dir_path, cat)
            for img_file in os.listdir(next_path):
                if os.path.splitext(img_file)[-1] in IMG_FORMATS:
                    img_path = os.path.join(next_path, img_file)
                    dataset.append((img_path, pid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, pid_container, num_imgs


def Init_dataset(dir, mode):
    return Dataset(dataset_dir=dir, mode=mode)
