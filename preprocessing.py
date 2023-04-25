import os
import pickle
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import preprocess_input

def tf_load_images(dataset_dir, batch_size, img_size):
    train = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='training',
    )

    test = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='validation',
    )

    train = train.map(
        lambda x, y: (preprocess_input(x), y))
    test = test.map(
        lambda x, y: (preprocess_input(x), y))

    return train, test

#https://stackoverflow.com/questions/70260531/how-to-attach-or-get-filenames-from-mapdataset-from-image-dataset-from-directory

def load_images(dataset_dir="./data/kaggle"):
    images = []
    labels = []
    countries = tqdm(os.listdir(dataset_dir))
    for country in countries:
        countries.set_description(country)
        country_dir = os.path.join(dataset_dir, country)
        if os.path.isdir(country_dir):
            files = tqdm(os.listdir(country_dir), leave=False)
            for file in files:
                files.set_description(file)
                img_path = os.path.join(country_dir, file)
                with Image.open(img_path) as img:
                    img_array = np.array(img.resize((256,256)))
                    images.append(extract_embedding(img_array))
                    labels.append(country)
   
    num_images = len(images)
    imgs_and_labels = list(zip(images, labels))
    random.seed(0)
    random.shuffle(imgs_and_labels)
    shuffled_images, shuffled_labels = zip(*imgs_and_labels)
    shuffled_images, shuffled_labels = list(shuffled_images), list(shuffled_labels)
    
    split_index = int(0.8 * num_images)
    train_image_features = shuffled_images[:split_index]
    test_image_features = shuffled_images[split_index:]
    train_labels = shuffled_labels[:split_index]
    test_labels = shuffled_labels[split_index:]

    # train_image_features = images[:split_index]
    # test_image_features = images[split_index:]
    # train_labels = labels[:split_index]
    # test_labels = labels[split_index:]
    
    return dict(
        train_imgs    = train_image_features,
        test_imgs     = test_image_features,
        train_labels            = train_labels,
        test_labels             = test_labels
    )


def extract_embedding(imgs, model="resnet"):
    if model == "resnet":
        return preprocess_input(imgs)
    else:
        pass

def get_data(data_folder="./data/kaggle"):
    with open(os.path.join(data_folder, 'data.p'), 'rb') as data_file:
        data_dict = pickle.load(data_file)
        
    train_imgs = list(data_dict['train_imgs'])
    test_imgs = list(data_dict['test_imgs'])
    train_labels = list(data_dict['train_labels'])
    test_labels = list(data_dict['test_labels'])
    return (train_imgs, train_labels), (test_imgs, test_labels)

def create_pickle(data_folder="./data/kaggle"):
    with open(os.path.join(data_folder, 'data.p'), 'wb') as pickle_file:
        pickle.dump(load_images(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')

def one_hot_encode(labels, num_countries):
    layer = tf.keras.layers.Hashing(num_bins=num_countries, output_mode='one_hot')
    return layer(labels)

def main():
    create_pickle("./data/kaggle")
    pass

if __name__ == '__main__':
    main()