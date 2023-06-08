import sys
sys.path.append( '../')

from preprocessing.constants import SIGN_OF_PATCH_PREDICTION_DATA_FOLDER, RANK_OF_PATCH_PREDICTION_DATA_FOLDER, HYPERPARAMETER_TUNING_FOLDER, ARTIFACTS_FOLDER
from models.general_pre_training import general_pre_training

from hyperparameter_tuning.general_pre_training import oGetArchitectureTuner

from models.bootstrap_aggregation import bootstrap_aggregation

import os

from tensorflow.keras.models import load_model

def aGetRepresentationModels(sRepresentationName):
    aModels = []
    for root, dirs, files in os.walk(ARTIFACTS_FOLDER):
        for sFolderName in dirs:
            
            if sFolderName == sRepresentationName:
                sFoundFolderPath = (os.path.join(root, sFolderName))
                
                sBatchName = os.path.basename(root)
                
                if 'final_model' in os.listdir(sFoundFolderPath):
                    
                    oModel = load_model(
                        filepath = f'{sFoundFolderPath}\\final_model'
                    )
                    
                    aModels.append( oModel)
                    
    return aModels
                    

if __name__ == '__main__':
    
    sDatasetName = 'dist'
    sRepresentationName = f'{sDatasetName.title()[:3]}ERT'
    
    aRepresentationModels = aGetRepresentationModels(sRepresentationName)
    
    oAggregationModel = bootstrap_aggregation(aRepresentationModels)
    
    
    
    X_spp = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
    Y_spp = np.load(f'{SIGN_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')

    X_rpp = np.load(f'{RANK_OF_PATCH_PREDICTION_DATA_FOLDER}\\X_{sDatasetName}.npy')    
    Y_rpp = np.load(f'{RANK_OF_PATCH_PREDICTION_DATA_FOLDER}\\Y_{sDatasetName}.npy')


    dataset = Dataset.from_tensor_slices((X_spp, Y_spp, X_rpp, Y_rpp))
    train_dataset, validation_dataset = split_dataset(
        dataset,
        right_size = TEST_SIZE,
        shuffle = False
    )
    
    
    train_dataset = train_dataset.batch(len(validation_dataset)).get_single_element()
    validation_dataset = validation_dataset.batch(len(validation_dataset)).get_single_element()
    
    
    X_spp_train, Y_spp_train, X_rpp_train, Y_rpp_train = train_dataset
    X_spp_val, Y_spp_val, X_rpp_val, Y_rpp_val = validation_dataset
    
    