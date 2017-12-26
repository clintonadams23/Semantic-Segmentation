import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, reg_scale=1.0):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
    output_layer7 = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))

    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01)
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out_scaled, num_classes, 1, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
    skip_conn_4 = tf.add(output_layer7, layer4_1x1)
    output_layer4 = tf.layers.conv2d_transpose(skip_conn_4, num_classes, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))

    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001)
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out_scaled, num_classes, 1, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
    skip_conn3 = tf.add(output_layer4, layer3_1x1)
    output_layer3 = tf.layers.conv2d_transpose(skip_conn3, num_classes, 16, 8, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale))
    
    return output_layer3
tests.test_layers(layers)


def optimize(last_layer, correct_label, learning_rate, num_classes, reg_scale=1.0):
    """
    Build the TensorFLow loss and optimizer operations.
    :param last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = cross_entropy_loss + reg_scale * sum(reg_losses)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, keep_prob_value = 1, learning_rate_value = 0.1):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        for image, label in get_batches_fn(batch_size):
            feed_dict = {
                input_image: image,
                correct_label: label,
                keep_prob: keep_prob_value,
                learning_rate: learning_rate_value
            }
            _, batch_loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            total_loss += batch_loss
            batch_count += 1
        
        loss = total_loss / batch_count
        print('epoch {}: loss={}'.format(epoch, loss))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    EPOCHS = 15
    BATCH_SIZE = 10
    REG_SCALE = 1e-3
    LEARNING_RATE = 1e-4
    KEEP_PROB = 0.5

    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        input_image, keep_prob, layer_3_out, layer_4_out, layer_7_out = load_vgg(sess, vgg_path)
        last_layer = layers(layer_3_out, layer_4_out, layer_7_out, num_classes, REG_SCALE)
        logits, train_op, loss = optimize(last_layer, correct_label, learning_rate, num_classes, REG_SCALE)

        sess.run(tf.global_variables_initializer())
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, loss, input_image, correct_label, keep_prob, learning_rate, KEEP_PROB, LEARNING_RATE)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
