import tensorflow as tf
import numpy as np

def generate_heatmap(model, image):
    # build model from inputs to output of last convolution layer
    last_conv_layer = model.layers[0].get_layer("conv5_block3_out")
    last_conv_layer_model = tf.keras.Model(model.layers[0].inputs, last_conv_layer.output)

    # build model from output of last conv layer to final predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_ind in [-3, -2, -1]:
        x = model.layers[0].layers[layer_ind](x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # get gradients of prediction wrt conv output
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(image)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
        
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # multiply gradients by its feature map
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
        
    # Average over all the filters to get a single 2D array
    gradcam = np.mean(last_conv_layer_output, axis=-1)
    # Clip the values (equivalent to applying ReLU) and normalize values
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)

    return gradcam