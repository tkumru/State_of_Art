import tensorflow as tf
from scipy.io import loadmat
import numpy as np

class Load_VGG:
    image_height = 600
    image_width = 800
    color_channels = 3


    def __init__(self, model_path='bin_art/cnn_model/imagenet-vgg-verydeep-19.mat') -> None:
        self.model_path = model_path

        self.vgg = loadmat(self.model_path)
        self.vgg_layers = self.vgg['layers']

    def main(self):
        graph =dict()
        graph['input']   = tf.Variable(np.zeros((1, Load_VGG.image_height, Load_VGG.image_width, Load_VGG.color_channels)), dtype = 'float32')
        graph['conv1_1']  = self.conv2d_relu(graph['input'], 0, 'conv1_1')
        graph['conv1_2']  = self.conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = self.avg_pooling(graph['conv1_2'])
        graph['conv2_1']  = self.conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2']  = self.conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = self.avg_pooling(graph['conv2_2'])
        graph['conv3_1']  = self.conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2']  = self.conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3']  = self.conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4']  = self.conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = self.avg_pooling(graph['conv3_4'])
        graph['conv4_1']  = self.conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2']  = self.conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3']  = self.conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4']  = self.conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = self.avg_pooling(graph['conv4_4'])
        graph['conv5_1']  = self.conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2']  = self.conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3']  = self.conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4']  = self.conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = self.avg_pooling(graph['conv5_4'])

        return graph
    
    def weight(self, layer, expected_layer_name):
        vgg_layer = self.vgg_layers

        weight_bias = vgg_layer[0][layer][0][0][2]
        w = weight_bias[0][0]
        b = weight_bias[0][1]
        layer_name = vgg_layer[0][layer][0][0][0][0]

        assert(layer_name == expected_layer_name)

        return w, b

    def relu(self, conv2d_layer):
        return tf.nn.relu(conv2d_layer)

    def conv2d(self, prev_layer, layer, layer_name):
        weight, bias = self.weight(layer, layer_name)

        weight = tf.constant(weight)
        bias = tf.constant(np.reshape(bias, (bias.size)))

        return tf.nn.conv2d(prev_layer, filters=weight, strides=[1, 1, 1, 1], padding='SAME') + bias
    
    def conv2d_relu(self, prev_layer, layer, layer_name):
        return self.relu(self.conv2d(prev_layer, layer, layer_name))

    def avg_pooling(self, prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
