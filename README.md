## How to Run the Code

1. Install Poetry by running ```curl -sSL https://install.python-poetry.org | python3 -```.
2. Install the project dependencies by running ```poetry install```.
3. Run the model training/testing script by running ```poetry run main```.
    - By not specifying any arguments, the script will only test the default model.
    - To train the model, specify the flag ```--train```.
    - To specify the number of epochs, specify the flag ```--epochs <number>```.
    - To specify the learning rate, specify the flag ```--lr <number>```.
    - To specify the model, specify the flag ```--model <model_name>```.
      - By default, the model loaded is ```Franzin/xlm-roberta-base-goemotions-ekman-multilabel```
      - Valid models are:
        1. ```Franzin/xlm-roberta-base-goemotions-ekman-multilabel```
        2. ```Franzin/roberta-base-goemotions-ekman-multilabel```
        3. ```Franzin/bigbird-roberta-base-goemotions-ekman-multilabel```
        4. ```Franzin/xlm-roberta-base-goemotions-ekman-multiclass```
        5. ```Franzin/roberta-base-goemotions-ekman-multiclass```
        6. ```Franzin/bigbird-roberta-base-goemotions-ekman-multiclass```
    - To specify the batch size, specify the flag ```--batch_size <number>```.
    - To specify the dataset, specify the flag ```--dataset <dataset_name>```.
      - By default, the dataset loaded is ```goemotions```
      - Valid datasets are:
        1. ```goemotions```
        2. ```meld```
4. Run the webserver script by running ```poetry run webserver```.
    - By default, the webserver will run on ```http://localhost:5000```.