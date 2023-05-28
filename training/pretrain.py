import constants as train_c

from models.pre_training import Pre_Training

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def Pretrain(X_train, Y_train, sTaskType ,sArtifactsFolder, fLearningRate, oLoss, oMetrics):
    oModel = Pre_Training(sTaskType = sTaskType)
    
    oModel.compile(
        loss = oLoss, 
        metrics = oMetrics,
        optimizer= Adam(
            learning_rate=ExponentialDecay(
                initial_learning_rate=fLearningRate,
                decay_steps=10**2,
                decay_rate=0.9
            )
        )
    )
    
    oModel.fit(
        X_train, 
        Y_train, 
        epochs= train_c.NR_OF_EPOCHS, 
        batch_size=train_c.BATCH_SIZE, 
        verbose=1
    )
    
    
    oModel.save_weights(
        f'{sArtifactsFolder}\\{sTaskType}\\',
        save_format ='tf'
    )