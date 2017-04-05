import os
import sys
import tarfile
import random
import pickle
import numpy as np
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from collections import Counter


url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_dir = './data'

num_classes = 10
image_size = 28
pixel_depth = 255.0

train_size = 200000
valid_size = 10000
test_size = 18724

np.random.seed(133)


def download_progress_hook(count, blockSize, totalSize):
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

    last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """notMNISTデータをダウンロードしてサイズを検証"""
    # notMNISTをダウンロード
    dest_filename = os.path.join(data_dir, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download: ', filename)
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')

    # ファイルサイズを検証
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception('Failed to verify ' + dest_filename)

    return dest_filename


def maybe_extract(filename, force=False):
    """tar.gzを解凍"""
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove tar.gz

    # 解凍
    if os.path.isdir(root) and not force:
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_dir)
        tar.close()

    data_dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))
                 if os.path.isdir(os.path.join(root, d))]
    if len(data_dirs) != num_classes:
        raise Exception('Excepted %d directories, one per class. Found %d instead.' % (
            num_classes, len(data_dir)))

    return data_dirs


def draw_images(root_dir, out_path):
    """各クラスの画像サンプルを描画"""
    assert len(root_dir) == num_classes  # A to J
    num_cols = 10
    pos = 1
    for i in range(num_classes):
        target_dir = root_dir[i]
        for j in range(num_cols):
            plt.subplot(num_classes, num_cols, pos)
            random_file = random.choice(os.listdir(target_dir))
            image = scipy.misc.imread(os.path.join(target_dir, random_file))
            plt.imshow(image, cmap=plt.get_cmap('gray'))
            plt.axis('off')
            pos += 1
    plt.savefig(out_path)


def load_letter(letter_dir, min_num_images):
    """各アルファベットのデータをロード"""
    image_files = os.listdir(letter_dir)
    # (num image, image width, image height)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    image_index = 0
    print(letter_dir)
    for image in image_files:
        image_file = os.path.join(letter_dir, image)
        try:
            # 画素値を[-0.5, 0.5]に正規化
            image_data = (scipy.ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
            image_index += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, "- it's ok, skipping.")

    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))

    return dataset


def maybe_pickle(data_dirs, min_num_images_per_class, force=False):
    """Pickle dataset"""
    dataset_names = []
    for d in data_dirs:
        set_filename = d + '.pickle'
        dataset_names.append(set_filename)

        if os.path.exists(set_filename) and not force:
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(d, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def draw_datasets(pickle_file, out_path):
    """pickleファイルからデータを取り出して描画"""
    print(pickle_file)
    with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        print(letter_set.shape)
    num_rows = num_cols = 10
    pos = 1
    for i in range(num_rows):
        for j in range(num_cols):
            plt.subplot(num_rows, num_cols, pos)
            image = letter_set[random.randint(0, len(letter_set))]
            plt.imshow(image, cmap=plt.get_cmap('gray'))
            plt.axis('off')
            pos += 1
    plt.savefig(out_path)


def verify_datasets(datasets):
    """データ数を検証"""
    for pickle_file in datasets:
        with open(pickle_file, 'rb') as f:
            letter_set = pickle.load(f)
        print("%s: %d" % (pickle_file, len(letter_set)))


def make_arrays(nb_rows, img_size):
    """空のデータ配列を作成して返す"""
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)

    # 空の配列を作成
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


def randomize(dataset, labels):
    """データをシャッフル"""
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def verify_class_balance(labels):
    """各クラスのデータ数をカウント"""
    count_dict = Counter(labels)
    for i in range(num_classes):
        print("class%d: %d" % (i, count_dict[i]))


def main():
    result_dir = 'result'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

    print('train_filename:', train_filename)
    print('test_filename:', test_filename)

    train_dirs = maybe_extract(train_filename)
    test_dirs = maybe_extract(test_filename)

    print('train_dirs:', train_dirs)
    print('test_dirs:', test_dirs)

    draw_images(train_dirs, os.path.join(result_dir, 'notmnist.png'))

    train_datasets = maybe_pickle(train_dirs, 45000)
    test_datasets = maybe_pickle(test_dirs, 1800)

    print('train_datasets:', train_datasets)
    print('test_datasets:', test_datasets)

    draw_datasets(train_datasets[0], os.path.join(result_dir, 'A.png'))
    verify_datasets(train_datasets)
    verify_datasets(test_datasets)

    # 訓練データと検証データに分割
    valid_dataset, valid_labels, train_dataset, train_labels = \
        merge_datasets(train_datasets, train_size, valid_size)

    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    # データをシャッフル
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)

    print('train:', train_dataset.shape, train_labels.shape)
    print('validate:', valid_dataset.shape, valid_labels.shape)
    print('test:', test_dataset.shape, test_labels.shape)

    # 各クラスのデータ数にばらつきがないか検証
    verify_class_balance(train_labels)
    verify_class_balance(valid_labels)
    verify_class_balance(test_labels)

    # 訓練データ、検証データ、テストデータを1つのpickleにまとめる
    pickle_file = os.path.join(data_dir, 'notMNIST.pickle')
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)


if __name__ == '__main__':
    main()
