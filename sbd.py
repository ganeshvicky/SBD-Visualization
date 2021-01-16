import os
import numpy as np
#import tensorflow as tf
from matplotlib import pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Params:
    F = 16
    L = 3 #sdcnn
    S = 2 #dcnn [4 cnn each]
    D = 256 #dense layer neurons
    INPUT_WIDTH = 48
    INPUT_HEIGHT = 27
    CHECKPOINT_PATH = None


class sbd:

    def __init__(self, params: Params, session=None):
        self.params = params
        self.session = session or tf.Session()
        self._build()
        self._restore()

    def _build(self):
        tempflag = True
        def shape_text(tensor):
            return ", ".join(["?" if i is None else str(i) for i in tensor.get_shape().as_list()])

        with self.session.graph.as_default():
            print("Creating Build")

            with tf.variable_scope("TransNet"):
                def conv3d(inp, filters, dilation_rate):
                    return tf.keras.layers.Conv3D(filters, kernel_size=3, dilation_rate=(dilation_rate, 1, 1),
                                                  padding="SAME", activation=tf.nn.relu, use_bias=True,
                                                  name="Conv3D_{:d}".format(dilation_rate))(inp)

                self.inputs = tf.placeholder(tf.uint8,
                                             shape=[None, None, self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH, 3])
                self.predictions = self.inputs
                net = tf.cast(self.inputs, dtype=tf.float32) / 255.
                print(" " * 10, "Input ({})".format(shape_text(net)))
                self.predictions = net

                for idx_l in range(self.params.L):
                    with tf.variable_scope("SDDCNN_{:d}".format(idx_l + 1)):
                        filters = (2 ** idx_l) * self.params.F
                        print(" " * 10, "SDDCNN_{:d}".format(idx_l + 1))

                        for idx_s in range(self.params.S):
                            with tf.variable_scope("DDCNN_{:d}".format(idx_s + 1)):
                                net = tf.identity(net)
                                
                                conv1 = conv3d(net, filters, 1)
                                # if(tempflag):
                                self.predictions = conv1
                                #     tempflag=False
                                conv2 = conv3d(net, filters, 2)
                                conv3 = conv3d(net, filters, 4)
                                
                                conv4 = conv3d(net, filters, 8)
                                net = tf.concat([conv1, conv2, conv3, conv4], axis=4)
                                
                                print(" " * 10, "> DDCNN_{:d} ({})".format(idx_s + 1, shape_text(net)))

                        net = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(net)
                        
                        print(" " * 10, "MaxPool ({})".format(shape_text(net)))

                shape = [tf.shape(net)[0], tf.shape(net)[1], np.prod(net.get_shape().as_list()[2:])]
                net = tf.reshape(net, shape=shape, name="flatten_3d")
                print(" " * 10, "Flatten ({})".format(shape_text(net)))
                net = tf.keras.layers.Dense(self.params.D, activation=tf.nn.relu)(net)
                print(" " * 10, "Dense ({})".format(shape_text(net)))

                self.logits = tf.keras.layers.Dense(2, activation=None)(net)
                print(" " * 10, "Logits ({})".format(shape_text(self.logits)))
                #self.predictions = tf.nn.softmax(self.logits, name="predictions")[:, :, 1]
                print(" " * 10, "Predictions ({})".format(shape_text(self.predictions)))

            print("Network built.")
            no_params = np.sum([int(np.prod(v.get_shape().as_list())) for v in tf.trainable_variables()])
            print("Found {:d} trainable parameters.".format(no_params))


    def _restore(self):
        if self.params.CHECKPOINT_PATH is not None:
            saver = tf.train.Saver()
            saver.restore(self.session, self.params.CHECKPOINT_PATH)
            print("Parameters restored from '{}'.".format(os.path.basename(self.params.CHECKPOINT_PATH)))

    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and \
               list(frames.shape[2:]) == [self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH, 3],\
            "Input shape must be [batch, frames, height, width, 3]."
        return self.session.run(self.predictions, feed_dict={self.inputs: frames})

    def predict_video(self, frames: np.ndarray):
        assert len(frames.shape) == 4 and \
               list(frames.shape[1:]) == [self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH, 3], \
            "Input shape must be [frames, height, width, 3]."



        def input_iterator():

            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

            #print(no_padded_frames_end)

            start_frame = np.expand_dims(frames[0], 0)
            # print("start frames", start_frame.shape)
            # print()
            end_frame = np.expand_dims(frames[-1], 0) 
            # print("end frames", end_frame.shape)
            #print("test", start_frame.shape)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            #print("len", padded_inputs.shape)

            # print("total padded frames:", padded_inputs.shape)
            # print(padded_inputs[2500,:,:,0].shape)
            # plt.imshow(padded_inputs[2500,:,:,0])
            # plt.show()
            # print(padded_inputs[2501,:,:,0].shape)
            # plt.imshow(padded_inputs[2501,:,:,0])
            # plt.show()
            # for i in range(2500, 2505):
            #     print(padded_inputs[i,:,:,0])

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr+100]
                ptr += 50
                yield out

        res = []
        for inp in input_iterator():
            # print(inp)
            # for i in range(100):
            #     plt.imshow(inp[i, :, :, 0])
            #     plt.show()
            pred = self.predict_raw(np.expand_dims(inp, 0))[0, 25:75]
            #print(pred.shape)
            res.append(pred)  #0.2, 
            #print(res)

            print("\rProcessing video frames {}/{}".format(
                min(len(res) * 50, len(frames)), len(frames)
            ), end="")
        print("")
        return np.concatenate(res)[:len(frames)]