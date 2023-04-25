from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from preprocessing import get_data, one_hot_encode, tf_load_images

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
        self.model.add(tf.keras.layers.Dense(500, activation='relu'))
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    def call(self, inputs):
        return self.model(inputs)

def parseArguments():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--load_weights", action="store_true") # load weights from most recent checkpoint
    parser.add_argument("--heatmap", action="store_true") # generate and save heatmap
    parser.add_argument("--batch_size", type=int, default=100) # batch size
    parser.add_argument("--num_epochs", type=int, default=2) # epochs
    parser.add_argument("--input_dim", type=int, default=256) # input image dimension
    parser.add_argument("--learning_rate", type=float, default=1e-3) # learning rate
    parser.add_argument("--num_classes", type=int, default=124) # number of classes (countries
    parser.add_argument("--data_dir", type=str, default="./data") # directory of dataset folders
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints") # directory of checkpoints
    parser.add_argument("--heatmap_dir", type=str, default="./heatmaps") # directory of heatmaps
    args = parser.parse_args()
    return args

def main(args):

    # (train_imgs, train_labels), (test_imgs, test_labels) = get_data()
    # unique_words = sorted(set(train_labels + test_labels))
    # vocabulary = {w:i for i, w in enumerate(unique_words)}

    # # Vectorize, and return output tuple.
    # train_labels = np.array(list(map(lambda x: vocabulary[x], train_labels)))
    # test_labels  = np.array(list(map(lambda x: vocabulary[x], test_labels)))
    # train_imgs = np.array(train_imgs)
    # test_imgs = np.array(test_imgs)

    train, test = tf_load_images(args.data_dir + "/kaggle", args.batch_size, args.input_dim)
    
    model = CountryClassifier(num_classes=args.num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy'],
    )

    # set up checkpoint callback
    checkpoint_path = args.checkpoint_dir + "simple_train/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # load weights from most recent checkpoint
    if args.load_weights:
        model.load_weights(tf.train.latest_checkpoint(args.checkpoint_dir + "simple_train"))
    
    # only save weights if not loading from checkpoint
    callbacks = [] if args.load_weights else [cp_callback]

    # train model
    model.fit(
        train,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=(test),
        callbacks=callbacks
    )

    # print()
    # print(model.summary())
if __name__ == '__main__':
    args = parseArguments()
    main(args)
