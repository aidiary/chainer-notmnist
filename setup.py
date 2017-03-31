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

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = './data'

num_classes = 10
image_size = 28
pixel_depth = 255.0

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
    dest_filename = os.path.join(data_root, filename)
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
        tar.extractall(data_root)
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


def main():
    result_dir = 'results'

    if not os.path.exists(data_root):
        os.makedirs(data_root)

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


if __name__ == '__main__':
    main()
