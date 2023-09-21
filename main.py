import argparse
import os
import pickle
from datetime import datetime
from sys import getsizeof

import cv2
import numpy as np
from tqdm import tqdm

from Data import Dataset
from Feature import FeatureExtractor
from Metrics import Stats

# Constants
k = 20
path = ".\\images"
output = ".\\output"
semantic = ".\\feature.pkl"
seman_name = ".\\names.pkl"

# Functions


def search(feature_list, query_feat, top, names):
    result = {}
    for element in tqdm(range(len(feature_list))):
        d = distance(feature_list[element], query_feat)
        result[names[element]] = d
    result = sorted(result.items(), key=lambda x: x[1])
    return result[:top]


def ReadImage(string):
    img = cv2.imread(string, cv2.IMREAD_COLOR)
    return img


def DescriptorProcessing(path, fe):
    images = []
    names = []
    for filename in tqdm(os.listdir(path)):
        direct = os.path.join(path, filename)
        img = ReadImage(direct)
        if img is not None:
            des = fe.extract(img)
            images.append(des)
            names.append(filename)
    return images, names


def distance(x, y):
    return np.linalg.norm(x - y)


def SaveHandling(finale, output_dir):
    for d in finale:
        obj, val = d[0], d[1]
        name = str(obj)
        print('Img ' + name + ' has distance value = ' + str(val))
        source = os.path.join(output_dir, name)
        ref = os.path.join(path, name)
        cv2.imwrite(source, ReadImage(ref))


def ClearAllFolder(folder):
    for files in os.listdir(folder):
        os.remove(os.path.join(folder, files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='image to query by path')
    args = parser.parse_args()
    q_path = args.input

    data = Dataset("groundtruth.json")
    print(data.getInfo())
    ClearAllFolder(output)

    start = datetime.now()
    # test descriptor class
    fe = FeatureExtractor()

    feature_list = None
    names = None
    query_img = ReadImage(q_path)
    feat_query = fe.extract(query_img)

    if os.path.isfile(semantic) is not True or os.path.isfile(seman_name) is not True:
        begin = datetime.now()
        feature_list, names = DescriptorProcessing(path, fe)
        pickle.dump(feature_list, open("feature.pkl", "wb"))
        pickle.dump(names, open("names.pkl", "wb"))
        end = datetime.now()
        print(f"Times to create *.pkl files: {end - begin}")

    if feature_list is None or names is None:
        feature_list = pickle.load(open("feature.pkl", "rb"))
        names = pickle.load(open("names.pkl", "rb"))

    result = search(feature_list, feat_query, k, names)
    SaveHandling(result, output)
    end = datetime.now()

    file_quer = os.path.basename(q_path)
    value = data.getDataFrame()
    score = Stats(value)
    time = end - start
    size_mem = getsizeof(feature_list) + getsizeof(names)
    score.access(file_quer)
    score.scoring(list(dict(result).keys()))
    score.saveStats(size_mem, time)