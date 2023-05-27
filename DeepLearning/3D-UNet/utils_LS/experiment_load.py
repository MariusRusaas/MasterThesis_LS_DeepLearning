import json
import os

def load_experiment_results(ex_path, num_models, filename):
    ''' Function that loads the results from a RayTune experiment.

    Parameters:
    - ex_path (str):    RayTune folder path
    - num_models (int): Number of models that was attempted
    - filename (str):   Experiment folder name

    Returns:
    - Tensor: Torch tensor with one LS volume
    '''
    results = []

    ex_content = os.listdir(ex_path)

    for i in range(1, num_models+1):
        this_model = {}
        model_out = ex_content[i]
        model_results_path = os.path.join(ex_path, model_out)
        model_results_json = model_results_path + '/' + filename

        iteration = []
        val_loss = []
        train_loss = []
        val_hist = []

        with open(model_results_json) as json_file:
            json_data = [json.loads(line) for line in json_file]
            for epoch in range(len(json_data)):
                val_loss.append(json_data[epoch]['val_loss'])
                train_loss.append(json_data[epoch]['train_loss'])
                val_hist.append(json_data[epoch]['hist_val'])
                iteration.append(json_data[epoch]['training_iteration'])
        this_model['val_loss'] = val_loss
        this_model['train_loss'] = train_loss
        this_model['val_hist'] = val_hist
        this_model['iteration'] = iteration
        this_model['config'] = json_data[-1]['config']

        results.append(this_model)
    return results

