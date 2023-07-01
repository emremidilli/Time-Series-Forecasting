# import sys
# sys.path.append( '../')

# from preprocessing.constants import MASKED_PATCH_PREDICTION_DATA_FOLDER, HYPERPARAMETER_TUNING_FOLDER, ARTIFACTS_FOLDER

# from models.general_pre_training import masked_patch_prediction,Train

# # from hyperparameter_tuning.general_pre_training import oGetArchitectureTuner, oGetOptimizerTuners

# from tensorflow.data import Dataset
# from tensorflow.keras.utils import split_dataset

# from training.constants import *
# import numpy as np
# import os
# import shutil


#     sOptimumHyperparametersFolder = f'{HYPERPARAMETER_TUNING_FOLDER}\\{sRepresentationName}'

#     oTunerArchitecture = oGetArchitectureTuner(
#         sLogsFolder = sOptimumHyperparametersFolder
#     )

    
#     oBestArchitecture = oTunerArchitecture.get_best_hyperparameters(1)[0]

#     nr_of_encoder_blocks = oBestArchitecture.get('nr_of_encoder_blocks')
#     nr_of_heads = oBestArchitecture.get('nr_of_heads')
#     dropout_rate = oBestArchitecture.get('dropout_rate')
#     nr_of_ffn_units_of_encoder = oBestArchitecture.get('nr_of_ffn_units_of_encoder')
#     embedding_dims = oBestArchitecture.get('embedding_dims')
    
#     _ , oTunerMpp = oGetOptimizerTuners(
#                 sLogsFolder  = sOptimumHyperparametersFolder,
#                 nr_of_encoder_blocks = nr_of_encoder_blocks, 
#                 nr_of_heads = nr_of_heads, 
#                 dropout_rate = dropout_rate, 
#                 nr_of_ffn_units_of_encoder = nr_of_ffn_units_of_encoder, 
#                 embedding_dims = embedding_dims
#             )

#     oBestNpp = oTunerNpp.get_best_hyperparameters(1)[0]
#     oBestMpp = oTunerMpp.get_best_hyperparameters(1)[0]


if __name__ == '__main__':
    
#     sDatasetName = sys.argv[1] # ['dist', 'tic', 'tre', 'sea', 'known', 'observed']
#     iNrOfEpochs = int(sys.argv[2])

#     sRepresentationName = f'{sDatasetName.title()[:3]}ERT'
    
#     # load datasets
#     X_mpp = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
#     Y_mpp = np.load(f'{MASKED_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')


#     # delete previously created artifacts
#     sArtifactsFolder = f'{ARTIFACTS_FOLDER}\\GPreT\\{sRepresentationName}'
#     if os.path.exists(sArtifactsFolder) == True:
#         shutil.rmtree(sArtifactsFolder)

    
#     # get hyperparameters
#     nr_of_encoder_blocks = 4
#     nr_of_heads = 6
#     dropout_rate = 0.01
#     nr_of_ffn_units_of_encoder = 128
#     embedding_dims = 64
    
#     fLearningRate = 1e-4
#     fMomentumRate = 0.85
    
#     # build model
#     oModelMpp = masked_patch_prediction(
#         iNrOfChannels = 3,
#         iNrOfQuantiles = 3,
#         iNrOfLookbackPatches = 16,
#         iNrOfForecastPatches = 4,
#         iNrOfEncoderBlocks = nr_of_encoder_blocks,
#         iNrOfHeads = nr_of_heads,
#         iContextSize = X_mpp.shape[-1],
#         fDropoutRate = dropout_rate, 
#         iEncoderFfnUnits = nr_of_ffn_units_of_encoder,
#         iEmbeddingDims = embedding_dims
#     )
    
#     # train & save the model
#     Train(
#         oModelMpp, 
#         X_mpp, 
#         Y_mpp, 
#         f'{sArtifactsFolder}\\mpp',
#         fLearningRate,
#         fMomentumRate ,
#         iNrOfEpochs, 
#         MINI_BATCH_SIZE
#     )
    

