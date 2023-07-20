# Import argument parser
import argparse
# Import modules from same directory
from modules import ML_Pipeline

# Import available models and scalers from modules
models = {'Classification':ML_Pipeline().classification_models}

all_models = {type(item).__name__:item for sublist in models.values() for item in sublist}

scalers = ML_Pipeline().scalers

# Parser arguments
parser = argparse.ArgumentParser(description='Settings')

parser.add_argument('--db_path', default = './data/fishing.db', type=str,
                    help = "Database directory: default = './data/fishing.db'")

parser.add_argument('--scaler_type', default = 'StandardScaler', type=str,
                    help="scaler_type options: {}".format(list(scalers.keys())))

parser.add_argument('--model_selection', default = 'all', type=str,
                    help = "specific model_selection options: {}".format(list(all_models.keys())))

# Import parameters from parser
scaler = scalers[parser.parse_args().scaler_type]
model_selection = parser.parse_args().model_selection
db_path = parser.parse_args().db_path

# def read_n_process_data(scaler, model):
#     # Instantiate our model
#     mlp = ML_Pipeline(scaler=scaler, model_selection=model)
#     # Injest data
#     mlp.injest_data(from_path = db_path, verbose=False)
#     # Clean and pre-process data
#     mlp.pre_process(verbose=False)
#     # Split data into training and test sets
#     mlp.split(test_size=0.25, random_state=42, verbose=False)

#     return mlp


def evaluate_model(scaler, model):
    """
    Function to print model evaluation results based on appropriate metrics.
    Arguments:
    scaler: Scaler class to be used.
    model: Model class to be used.
    """
    # Instantiate our model
    mlp = ML_Pipeline(scaler=scaler, model_selection=model)
    # Injest data
    mlp.injest_data(from_path = db_path, verbose=False)
    # Clean and pre-process data
    mlp.pre_process(verbose=False)
    # Split data into training and test sets
    mlp.split(test_size=0.25, random_state=42, verbose=False)

    # Fit model to our training dataset
    mlp.fit(verbose=False)
    # Make predictions on our test dataset
    mlp.predict(x_test=None, verbose=False)
    # Print current model settings
    mlp.settings()
    # Evaluate and print metrics
    mlp.evaluate(y_test=None)

print('Model/s selected:', parser.parse_args().model_selection)
print('Scaler selected:', parser.parse_args().scaler_type)
print('DB file location:', parser.parse_args().db_path)

print("\nBegin evaluation")
print('-'*55)

# mlp = read_n_process_data()

for model_name, model in all_models.items():
    # Evaluate all models if model selection is not chosen
    if model_selection=='all':
        evaluate_model(scaler=scaler, model=model)
        print('-'*55)
    # Evaluate only the chosen model if model selection is specified
    elif model_selection==model_name:
        evaluate_model(scaler=scaler, model=model)
        print('-'*55)
print("End.")
        

    


