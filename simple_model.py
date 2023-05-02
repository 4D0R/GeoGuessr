from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from preprocessing import country_load, lat_long_load

class CountryClassifier(tf.keras.Model):

    def __init__(self, num_classes, input_shape=(256,256,3)):
        super(CountryClassifier, self).__init__()

        pretrained_model = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=input_shape,
            pooling='avg',classes=1000,
            weights='imagenet')
        for layer in pretrained_model.layers:
            layer.trainable=False

        dense1 = tf.keras.layers.Dense(500, activation='relu')(pretrained_model.layers[-1].output)
        dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')(dense1)
        self.model = tf.keras.Model(inputs=pretrained_model.inputs, outputs=dense2)


    def call(self, inputs):
        return self.model(inputs)

class CoordinateClassifier(tf.keras.Model):

    def __init__(self, input_shape=(256,256,3)):
        super(CoordinateClassifier, self).__init__()

        pretrained_model = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=input_shape,
            pooling='avg',classes=1000,
            weights='imagenet')
        for layer in pretrained_model.layers:
            layer.trainable=False

        dense1 = tf.keras.layers.Dense(500, activation='relu')(pretrained_model.layers[-1].output)
        dense2 = tf.keras.layers.Dense(2)(dense1)
        self.model = tf.keras.Model(inputs=pretrained_model.inputs, outputs=dense2)

    def call(self, inputs):
        return self.model(inputs)

def parseArguments():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--load_weights", action="store_true") # load weights from most recent checkpoint
    parser.add_argument("--heatmap", action="store_true") # generate and save heatmap
    parser.add_argument("--lat_long", action="store_true") # lat long model
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
    if args.lat_long:
        train, test = lat_long_load(args.data_dir + "/streetviews", args.batch_size, args.input_dim)
        model = CoordinateClassifier()
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        train, test = country_load(args.data_dir + "/streetviews", args.batch_size, args.input_dim)
        model = CountryClassifier(num_classes=args.num_classes)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), 
        loss=loss_fn, 
        metrics=['accuracy'],
    )

    # set up checkpoint callback
    folder = "/latlong/" if args.lat_long else "/country/"
    checkpoint_path = args.checkpoint_dir + folder + "cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # load weights from most recent checkpoint
    if args.load_weights:
        model.load_weights(tf.train.latest_checkpoint(args.checkpoint_dir + folder))
    
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

if __name__ == '__main__':
    args = parseArguments()
    main(args)
