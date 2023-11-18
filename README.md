# Folder Structure


```
.
├── 01_converted_data
│   └── dataset_name.csv
├── 02_formatted_data
│   └── model_id
│       ├── config.json
│       ├── fc_test.npy
│       ├── fc_train.npy
│       ├── ix_test.npy
│       ├── ix_train.npy
│       ├── lb_test.npy
│       ├── lb_train.npy
│       ├── ts_test.npy
│       └── ts_train.npy
└── 03_preprocessing
    └── model_id
        ├── dataset
        ├── input_preprocessor
        └── target_preprocessor
```