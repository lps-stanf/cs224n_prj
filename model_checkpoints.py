import os

import keras


class MyModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, dir, weights_name, epoch_period = 1):
        self.path_begin = os.path.join(dir, weights_name)
        self.period = max(1, epoch_period)
        self.log_file = open(os.path.join(dir, 'epoch.log'), 'w')

    def on_epoch_end(self, epoch, logs=None):
        # we want 1-indexed epoch in output
        epoch += 1
        self.log_file.write('{0}: {1}\n'.format(epoch, logs['loss']))

        if epoch % self.period == 0:
            filename = '{}_{:04}.hdf5'.format(self.path_begin, epoch)
            self.model.save_weights(filename)

    def on_train_end(self, logs=None):
        self.log_file.close()