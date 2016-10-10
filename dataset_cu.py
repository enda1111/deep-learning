try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import zipfile
import pickle
import os
import numpy as np
from PIL import Image


url_base = 'http://www.vision.cs.chubu.ac.jp/jointhog/SRC/'
key_file = 'dataset_training.zip'

dataset_dir = os.path.dirname(__file__)
save_file = os.path.join(dataset_dir, "cu.pkl")

train_num = 7000
test_num = 1000
img_dim = (3, 98, 66)
img_size = 6468


def load_cu(normalize=True, flatten=True, one_hot_label=False):
    """CUデータセットの読み込み

    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか

    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    if not os.path.exists(save_file):
        init_cu()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # if one_hot_label:
    #     dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
    #     dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])
    #
    # if not flatten:
    #     for key in ('train_img', 'test_img'):
    #         dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


def init_cu():
    download_cu()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def download_cu():
    _download(key_file)


def _download(file_name):
    file_path = os.path.join(dataset_dir, file_name)

    if os.path.exists(file_path):
        return
    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")


def _convert_numpy():
    image_array, label_array = _load_img(key_file)
    return _make_dataset(image_array, label_array)


def _make_dataset(images, labels):
    dataset = {}
    train_idx = np.random.choice(images.shape[0], train_num + test_num)
    dataset['train_img'] = images[train_idx[:train_num]]
    dataset['train_label'] = labels[train_idx[:train_num]]
    dataset['test_img'] = images[train_idx[train_num:]]
    dataset['test_label'] = labels[train_idx[train_num:]]
    return dataset


def _load_img(file_name):
    file_path = os.path.join(dataset_dir, file_name)
    pos_images = []
    neg_images = []
    with zipfile.ZipFile(file_path, 'r') as zip:
        for file_name in zip.namelist():
            if 'bmp' not in file_name:
                continue
            with zip.open(file_name) as file:
                image = Image.open(file, 'r')
                resize_image = image.resize((img_dim[1], img_dim[2]))
                image_array = np.asarray(resize_image).transpose(2, 0, 1)
                if 'Pos' in file_name:
                    pos_images.append(image_array)
                else:
                    neg_images.append(image_array)
    pos_image_array = np.array(pos_images)
    neg_image_array = np.array(neg_images)
    n, c, h, w = pos_image_array.shape
    image_array = np.zeros((len(pos_images + neg_images), c, h, w))
    image_array[0:len(pos_images)] = pos_image_array
    image_array[len(pos_images):] = neg_image_array

    label_array = np.ones((len(pos_images + neg_images)))
    label_array[len(pos_images):] = 0

    return image_array, label_array.astype(np.uint8)

if __name__ == '__main__':
    init_cu()
