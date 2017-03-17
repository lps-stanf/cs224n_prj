import os

import keras


class MyModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, dir, weights_name, epoch_period=1, model_id=None):
        self.path_begin = os.path.join(dir, weights_name)
        self.model_id = model_id
        self.period = max(1, epoch_period)
        self.log_file = open(os.path.join(dir, 'epoch_history_{0}.log'.format(self.model_id)), 'w')

    def on_epoch_end(self, epoch, logs=None):
        # we want 1-indexed epoch in output
        epoch += 1
        if 'val_loss' in logs:
            self.log_file.write('Epoch {0}: Train {1}, Val{2}\n'.format(epoch, logs['loss'], logs['val_loss']))
        else:
            self.log_file.write('Epoch {0}: Train {1}\n'.format(epoch, logs['loss']))

        if epoch % self.period == 0:
            filename = '{0}_{:04}_{2}.hdf5'.format(self.path_begin, epoch, self.model_id)
            self.model.save_weights(filename)

    def on_train_end(self, logs=None):
        self.log_file.close()