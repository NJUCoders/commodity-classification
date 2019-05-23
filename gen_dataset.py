import random
import sys
import threading
import time
from typing import List

import numpy as np
from PIL import Image
import csv
import pickle

image_size = (96, 96)


def gen_dataset(rows: List[List[object]], num: int):
    print(f"thread {num} is working...")
    pres = [lambda im: im, lambda im: im.rotate(90), lambda im:im.rotate(180), lambda im: im.rotate(270)]
    dataset = {'image': [], 'label': [], 'filename': []}
    begin = time.time()
    for i, row in enumerate(rows):
        print(f"{num}-{i}\t{time.time() - begin}")
        filename, label = row[0], row[1]
        with open(f'easy/data/{filename}', 'rb') as fimg:
            im = Image.open(fimg)  # 原始图像
            w = max(im.size)  # 正方形的宽度
            im = im.crop((0, 0, w, w)).resize(image_size)  # 补成正方形再压缩
            dataset['image'].append(np.asarray(random.choice(pres)(im)))
        dataset['label'].append(label)
        dataset['filename'].append(filename)
    with open(f'easy/train_set{str(image_size[0])}x{str(image_size[1])}padded_{str(num)}.pk', 'wb') as f:
        pickle.dump(dataset, f)
    del dataset
    print(f"thread {num} finished")


def gen_datasets():
    with open('easy/data.csv', 'r') as fcsv:
        reader = csv.reader(fcsv)
        header = next(reader)
        rows = [r for r in reader]
    threads = []
    thread_num = 40
    thread_split = int(100000/thread_num)  # 单个线程处理的数据量
    for i in range(thread_num):
        threads.append(threading.Thread(target=gen_dataset, args=(rows[i*thread_split:(i+1)*thread_split], i)))
        threads[-1].start()
    for tr in threads:
        tr.join()
    train_set = {'image': [], 'label': [], 'filename': []}
    for i in range(thread_num):
        with open(f'easy/train_set{str(image_size[0])}x{str(image_size[1])}padded_{str(i)}.pk', 'rb') as f:
            tmp_set = pickle.load(f)
        train_set['image'].extend(tmp_set['image'])
        train_set['label'].extend(tmp_set['label'])
        train_set['filename'].extend(tmp_set['filename'])
    with open(f'easy/train_set{str(image_size[0])}x{str(image_size[1])}padded.pk', 'wb') as f:
        pickle.dump(train_set, f)


if __name__ == '__main__':
    gen_datasets()
