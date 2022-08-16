# Scene Recognition
Classification of 365 Scenes by fine-tuning Vision Transformer on [Places365 Standard dataset.](http://places2.csail.mit.edu/)

## Example Scenes

| Images | Labels|
|--------|-------|
| ![baseball_field_00000517](https://user-images.githubusercontent.com/25534697/184756455-7ab798bb-2cd4-4907-8e8b-0c3060361d40.png)| Baseball Field|
| ![balcony-interior_00003593](https://user-images.githubusercontent.com/25534697/184755710-c8c113af-a5c7-43d6-9177-29132822f51a.png)| Balcony Interior|
|![embassy_00002301](https://user-images.githubusercontent.com/25534697/184756740-e6916a91-7f1b-4064-9e1c-821bc6b05b5c.png) | Embassy|
|![fire_escape_00000768](https://user-images.githubusercontent.com/25534697/184756768-c84e25ad-de66-44a4-92b6-4eeed9d706ca.png) | Fire Escape |
|![kitchen_00003271](https://user-images.githubusercontent.com/25534697/184756813-f8d3ef9d-a664-4e28-9b4f-d0c83bc2d711.png) | Kitchen |
|![lake-natural_00004559](https://user-images.githubusercontent.com/25534697/184756833-1ba4305f-33dc-4083-81ae-6968f249e0ef.png) | Lake Natural |
|![skyscraper_00004143](https://user-images.githubusercontent.com/25534697/184756915-62223fbd-dd21-437a-aa20-43e1f17eb0d8.png) | Skyscraper |
|![office_cubicles_00000716](https://user-images.githubusercontent.com/25534697/184757003-01c637b8-5eff-40c6-9032-ed7c395713e8.png) | Office Cubicles |
|![reception_00002637](https://user-images.githubusercontent.com/25534697/184757030-c9def29e-45d8-4f34-ba4d-6a9f0aefbcaa.png) | Reception |

## Result
Evaluation performed on the test set.

| Metric | Value |
|--------|-------|
|AUROC | **98.85** |
|Accuracy Top 5 | **81.87** |
|Accuracy Top 1 | **49.91** |
|F1-Score | **49.14** |
|Precision| **51.14** |
|Recall | **49.91** |

## How to Run
1. Download the [Places365 Standard dataset.](http://places2.csail.mit.edu/)
2. Update the paths of 
    - DATASET_TRAIN_PATH : Path of Places365 Standard training dataset.
    - DATASET_TEST_PATH : Path of Places365 Standard validation dataset.
    - DATASET_MAPPINGS_PATH : Path to store the dataset mappings for train and test datasets.
    - WANDB_PATH : Path to initialize Weights and Bias runs.
3.  Run preprocess_dataset.py to create mapping of images.
4.  Train the model by running train.py script.
5.  Evaluate the model on the test dataset by running evaluate.py script.

