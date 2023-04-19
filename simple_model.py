import numpy as np
import tensorflow as tf
from preprocessing import get_data, one_hot_encode

class CountryClassifier(tf.keras.Model):

    def __init__(self, num_classes, input_shape=(256,256,3)):
        super(CountryClassifier, self).__init__()

        self.model = tf.keras.Sequential()

        pretrained_model = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=input_shape,
            pooling='avg',classes=1000,
            weights='imagenet')
        for layer in pretrained_model.layers:
            layer.trainable=False

        self.model.add(pretrained_model)
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    def call(self, inputs):
        return self.model(inputs)

def main():
    NUM_COUNTRIES = 125

    (train_imgs, train_labels), (test_imgs, test_labels) = get_data()
    unique_words = sorted(set(train_labels + test_labels))
    vocabulary = {w:i for i, w in enumerate(unique_words)}

    # Vectorize, and return output tuple.
    train_labels = np.array(list(map(lambda x: vocabulary[x], train_labels)))
    test_labels  = np.array(list(map(lambda x: vocabulary[x], test_labels)))
    train_imgs = np.array(train_imgs)
    test_imgs = np.array(test_imgs)
    
    model = CountryClassifier(num_classes=NUM_COUNTRIES)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy'],
    )

    model.fit(
        train_imgs, train_labels,
        epochs=2,
        batch_size=100,
        validation_data=(test_imgs, test_labels)
    )

    # print()
    # print(model.summary())
if __name__ == '__main__':
    main()
