
class architecture_optimizer():
    '''Optimizes the architectural hyperparameters'''

    def __init__(
            self,
            training_dataset,
            validation_dataset,
            optimizer,
            params,
            boundaries):
        '''
        args:
            training_dataset (tf.data.Dataset)
            validation_dataset (tf.data.Dataset)
            optimizer (keras.optimizer)
        '''
        self.optimizer = optimizer

    def tune():
        '''Tunes the decision_variables'''
        ...


if __name__ == '__main__':
    '''
    There are 2 types of hyperamaeter tuning.
    1. Without changing the architecture
    2. With changing the architecture.

    As first step, we should identify what is the correct architecture.
    Later, we should set the optimizer hyperparameters.

    To set the architecture, we should use a central values of optimizers.
    '''
    ...
