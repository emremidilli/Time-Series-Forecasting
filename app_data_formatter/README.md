# Folder Structure
```
.
├── 01_converted_data
│   └── dataset_name.csv
├── 02_formatted_data
    └── model_id
        ├── config.json
        ├── fc_test.npy
        ├── fc_train.npy
        ├── ix_test.npy
        ├── ix_train.npy
        ├── lb_test.npy
        ├── lb_train.npy
        ├── ts_test.npy
        └── ts_train.npy
```

# Purpose

Reads a dataset from converted data folder.

The input dataset is in TimeSeriesDataset format of Pytorch with the columns of ['time_idx', 'group_id', 'value'].

This app creates set of datasets for a model_id in formatted data folder.