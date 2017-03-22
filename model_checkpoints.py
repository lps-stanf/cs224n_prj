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
            self.log_file.write('Epoch {0}\tTrain {1}\t Val {2}\n'.format(epoch, logs['loss'], logs['val_loss']))
        else:
            self.log_file.write('Epoch {0}\t Train {1}\n'.format(epoch, logs['loss']))

        if epoch % self.period == 0:
            filename = '{}_{:03}_{}.hdf5'.format(self.path_begin, epoch, self.model_id)
            self.model.save_weights(filename)

    def on_train_end(self, logs=None):
        self.log_file.close()


class BestModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, dir, model_name, epoch_period=1, model_id=None):
        self.model_id = model_id
        self.period = max(1, epoch_period)
        self.best_loss = 1.0e6
        self.best_epoch = -1
        self.model_name = model_name

        self.out_dir = os.path.join(dir, model_name)
        if not os.path.isdir(self.out_dir):
             os.makedirs(self.out_dir)
        
        self.log_file = open(os.path.join(dir, '{}__{}.log'.format(self.model_name, self.model_id)), 'w')

    def on_epoch_end(self, epoch, logs=None):
        # we want 1-indexed epoch in output
        epoch += 1
        if 'val_loss' in logs:
            if logs['val_loss'] < self.best_loss:
                self.best_loss = logs['val_loss']
                self.best_epoch = epoch
                print('\nA new best result! Val loss = {}\n'.format(logs['val_loss']))
                if epoch > self.period:
                    self.log_file.write('New best result achieved!\n')
                    filename = os.path.join(self.out_dir,
                                            'best_{}__{:03}__{}.hdf5'.format(self.model_name, epoch, self.model_id))
                    print('Saving weights\n')
                    self.model.save_weights(filename)
            self.log_file.write('Epoch {0}\tTrain {1}\t Val {2}\n'.format(epoch, logs['loss'], logs['val_loss']))

        else:
            self.log_file.write('Epoch {0}\t Train {1}\n'.format(epoch, logs['loss']))

        if epoch % self.period == 0:
            filename = os.path.join(self.out_dir, '{}__{:03}__{}.hdf5'.format(self.model_name, epoch, self.model_id))
            self.model.save_weights(filename)

    def on_train_end(self, logs=None):
        self.log_file.close()
