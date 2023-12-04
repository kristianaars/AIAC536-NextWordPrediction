# AIAC536-NextWordPrediction
Project work for AIAC536-NLP

## How to train the model
### 1. Prepare data
Dataset can be prepared using [data-preperation.ipynb](./data-preperation.ipynb). Please follow the notebook and change hyperparameters if needed.

### 2. Train the model
The file [train-model.ipynb](./train-model.ipynb) contains the code for formatting the dataset for training, as well as building the model, and perform training.
You will have to change the code if you wish to use your own prepared dataset.
The model currently uses LSTM, but this can easily be changed to GRU or other recurrent network models supported by tensorflow.

### 3. Run the API
Run `python api.py` to begin the api. Please note that the saved model and tokenizer file from step 2 must be specified in the appropriate variables inside [api.py](./api.py)