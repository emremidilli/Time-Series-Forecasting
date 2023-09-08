import gc

import os

import tensorflow as tf


class BaseCheckpointCallback(tf.keras.callbacks.Callback):
    '''Saves checkpoints for model, optimizer, epoch and step.'''
    def __init__(self, ckpt_dir, epoch_freq):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.epoch_freq = epoch_freq
        self.step_nr = 0

    def on_batch_end(self, batch, logs=None):
        '''Saves the batch number within an epoch'''
        self.step_nr = self.step_nr + 1


class PreTrainingCheckpointCallback(BaseCheckpointCallback):
    '''Checkpoint saver for pre-training model.'''

    def on_epoch_end(self, epoch, logs=None):
        '''Saves the model, optimizer, epoch and step.'''
        if epoch % self.epoch_freq == 0:
            self.epoch_nr = epoch

            checkpoint_epoch = tf.train.Checkpoint(
                epoch_nr=tf.Variable(self.epoch_nr, dtype=tf.int64),
                step_nr=tf.Variable(self.step_nr, dtype=tf.int64),
                model=self.model,
                mae_optimizer=self.model.mae_optimizer,
                cl_optimizer=self.model.cl_optimizer)

            checkpoint_epoch.save(
                file_prefix=self.ckpt_dir)

    def get_most_recent_ckpt(self, model, mae_optimizer, cl_optimizer):
        '''
        finds the latest checkpoint.
        returns epoch_nr, step_nr, model, mae_optimizer and cl_optimizer.
        '''
        ckpt_parent_dir = os.path.dirname(
            self.ckpt_dir)

        latest_ckpt = tf.train.latest_checkpoint(ckpt_parent_dir)

        if latest_ckpt:
            ckpt = tf.train.Checkpoint(
                epoch_nr=tf.Variable(0, dtype=tf.int64),
                step_nr=tf.Variable(0, dtype=tf.int64),
                model=model,
                mae_optimizer=mae_optimizer,
                cl_optimizer=cl_optimizer)

            ckpt.restore(latest_ckpt)
            epoch_nr = ckpt.epoch_nr.numpy()
            step_nr = ckpt.step_nr.numpy()
            model = ckpt.model
            mae_optimizer = ckpt.mae_optimizer
            cl_optimizer = ckpt.cl_optimizer

            self.step_nr = step_nr

        return epoch_nr, step_nr, model, mae_optimizer, cl_optimizer


class FineTuningCheckpointCallback(BaseCheckpointCallback):
    '''Checkpoint saver for fine-tuning model.'''

    def on_epoch_end(self, epoch, logs=None):
        '''Saves the model, optimizer, epoch and step.'''
        if epoch % self.epoch_freq == 0:
            self.epoch_nr = epoch

            checkpoint_epoch = tf.train.Checkpoint(
                epoch_nr=tf.Variable(self.epoch_nr, dtype=tf.int64),
                step_nr=tf.Variable(self.step_nr, dtype=tf.int64),
                model=self.model,
                optimizer=self.model.optimizer)

            checkpoint_epoch.save(
                file_prefix=self.ckpt_dir)

    def get_most_recent_ckpt(self, model, optimizer):
        '''
        finds the latest checkpoint.
        returns epoch_nr, step_nr, model, optimizer.
        '''
        ckpt_parent_dir = os.path.dirname(
            self.ckpt_dir)

        latest_ckpt = tf.train.latest_checkpoint(ckpt_parent_dir)

        if latest_ckpt:
            ckpt = tf.train.Checkpoint(
                epoch_nr=tf.Variable(0, dtype=tf.int64),
                step_nr=tf.Variable(0, dtype=tf.int64),
                model=model,
                optimizer=optimizer)

            ckpt.restore(latest_ckpt)
            epoch_nr = ckpt.epoch_nr.numpy()
            step_nr = ckpt.step_nr.numpy()
            model = ckpt.model
            optimizer = ckpt.optimizer

            self.step_nr = step_nr

        return epoch_nr, step_nr, model, optimizer


class LearningRateCallback(tf.keras.callbacks.Callback):
    '''Noam Learning rate schedule of "Attention is all you need paper"'''
    def __init__(
            self,
            d_model,
            warmup_steps=4000,
            scale_factor=1.0,
            remained_step_nr=0):
        super().__init__()

        self.d_model = tf.cast(d_model, dtype=tf.float32)

        self.warmup_steps = warmup_steps

        self.scale_factor = tf.cast(scale_factor, dtype=tf.float32)

        self.step_nr = remained_step_nr

    def schedule(self, step):
        '''calculates the new learning rate based on step (batch) number'''
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        unscaled = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        scaled = self.scale_factor * unscaled
        return scaled

    def on_batch_begin(self, batch, logs=None):
        '''sets the calculated learning rate to the optimzer'''
        self.step_nr = self.step_nr + 1
        lr = self.schedule(step=self.step_nr)
        tf.keras.backend.set_value(self.model.mae_optimizer.lr, lr)
        tf.keras.backend.set_value(self.model.cl_optimizer.lr, lr)


class RamCleaner(tf.keras.callbacks.Callback):
    '''Callback to clean RAM with garbage collector.'''

    def on_epoch_end(self, epoch, logs={}):
        '''
        Cleans the RAM after every epoch.
        '''
        gc.collect()
