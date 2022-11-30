[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7232440&assignment_repo_type=AssignmentRepo)
Change this README to describe how your code works and how to run it before you submit it on gradescope.

# Image Captioning
To change the model type between LSTM and Vanilla RNN, use the `default.json` and choose `model_type` for `LSTM` or `Vanilla`
To use Architectiure 2 of the model, simply use `arch2_model_factory.py` and `arch2_experiment.py` instead of `model_factory.py` and `experiment.py`

To train the image caption model, use:
```console
  $ python3 main.py default
```
or you can check `main.ipynb` for training procedures. 

In -model_factory.py, Encoder Class implemented a pretrained resnet50. And Decoder class implemented LSTM and Vanilla RNN network

In -experimemt.py, standard train and validation procedures are implemented.

To Generate image captioning results. Use `exp.test()` which is implemented in the test function of `Experiment class`. 
It automatically generates good and bad images using a temperature of 0.4. Also it generates captions generated by using 1) the deterministic
approach, and 2) using a very high and a very low temperature which is 5 and 0.001.


## Usage

* Define the configuration for your experiment. See `default.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment.py` based on the project requirements.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- arch2_model_factory.py: Factory to build models based on config Architecture 2 
- arch2_experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments For Architecture 2. 
- dataset_factory.py: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace
- main.ipynb main driver notebook 