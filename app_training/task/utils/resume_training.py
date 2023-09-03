from keras.callbacks import ModelCheckpoint

import os

import tensorflow as tf


class CustomModelCheckpoint(ModelCheckpoint):
    '''Custom ModelCheckpoint by epoch frequency.'''
    def __init__(self,
                 starting_epoch_checkpoint_dir,
                 epoch_freq,
                 **kwargs):
        super().__init__(**kwargs)

        self.starting_epoch_checkpoint_dir = os.path.join(
            starting_epoch_checkpoint_dir,
            'ckpt')

        self.filepath = os.path.join(
            self.filepath,
            'ckpt')

        self.epoch_freq = epoch_freq

    def on_epoch_end(self, epoch, logs=None):
        '''Saves model and the remained epoch to resume training.'''
        if epoch % self.epoch_freq == 0:

            checkpoint_epoch = tf.train.Checkpoint(
                starting_epoch=tf.Variable(epoch, dtype=tf.int64))

            checkpoint_epoch.save(
                file_prefix=self.starting_epoch_checkpoint_dir)

            return super().on_epoch_end(epoch, logs)

    def get_most_recent_weight_and_epoch_nr(self,
                                            model):
        '''
        finds the latest saved checkpoint
        assigns the weights of it to the model.
        returns and integer that indicates
            the latest remained epoch number.
        '''

        model_ckpt_parent_dir = os.path.dirname(self.filepath)
        starting_epoch_ckpt_parent_dir = os.path.dirname(
            self.starting_epoch_checkpoint_dir)

        latest_model_checkpoint = tf.train.latest_checkpoint(
            model_ckpt_parent_dir)
        if latest_model_checkpoint:
            model.load_weights(latest_model_checkpoint)

        latest_starting_epoch_checkpoint = tf.train.latest_checkpoint(
            starting_epoch_ckpt_parent_dir)

        if latest_starting_epoch_checkpoint:
            starting_epoch = 0
            checkpoint_starting_epoch = tf.train.Checkpoint(
                starting_epoch=tf.Variable(starting_epoch, dtype=tf.int64))

            checkpoint_starting_epoch.restore(latest_starting_epoch_checkpoint)
            starting_epoch = checkpoint_starting_epoch.starting_epoch.numpy()

        return starting_epoch


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''Learning rate schedule of "Attention is all you need paper"'''
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
