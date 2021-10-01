# dl_bio_example

The main purpose of this repository is to give examples how to train and evaluate neural networks using the [DLBio](https://github.com/pgruening/dlbio) package. In the file 'datasets/data_getter.py' you'll find information on how to create new datasets. 'models/model_getter.py' gives information about how to initialize models and create custom models.

The actual code for running and evaluating a single experiment is stored into a subfolder in the 'experiment' folder. The folder should contain:

* A [README.md](https://github.com/pgruening/dl_bio_example) file, describing the experiment. Including idea, results, and conclusion.
* A script 'exe_[...].py' to run the training processes. This script usually saves log.json files, containing, e.g., the training curves, opt.json that store information about the model and dataset, and the weights of a model ('model.pt'). 
* A python script to evaluate the results. Usually, based on the log files.

You can find an example for the training and evaluation of different model architectures on MNIST in the 'experiment/model_search'-folder.

## Setup

Run a terminal and follow the instructions below.

### Add the working directory to the PYTHONPATH

Most of the execute files are nested in the experiment folders. To import modules from the parent folder (e.g., helpers.py) you'll need to add the parent folder path to your PYTHONPATH.

On linux, open your .bashrc file in your home directory:
```
nano ~/.bashrc
```

Now add the path to the dlbio_example folder to your PYTHONPATH. To do this, just append this line (don't forget to write the actual path, not "your/path/to"):

```
export PYTHONPATH=your/path/to/dlbio_example:$PYTHONPATH
```

First press "control+o", then "control+x" to save your changes and exit nano. Finally, update the changes in your current terminal. You'll only need to do this once:

```
source ~/.bashrc
```
### Using virtual environments

One easy way to keep track of different packages with different version for different projects is a python virtual environment (VE).

To install virtualenv, run

```
sudo apt install virtualenv 
```

To facilitate the use of VEs, you can use the virtualenvwrapper package. Run:

```
pip install virtualenvwrapper
```

Again, open your .bashrc file with:
```
nano ~/.bashrc
```

In nano, append these lines to your file:

```
# virtualenv and virtualenvwrapper
export WORKON_HOME=/nfshome/gruening/.virtualenvs
source ~/.local/bin/virtualenvwrapper.sh
export PATH=$PATH:~/.local/bin
```

save and exit, then run:
```
source ~/.bashrc
```

At this point, you should be able to create a new VE with

```
mkvirtualenv dlbio_example -p python3
```

After running the command successfully, you should see a difference in your terminal: 
```
(dlbio_example) your_name@your_pc:
```

"(dlbio_example)" indicates that you are no in the new VE, called dlbio_example. When you run python, your version should be python3.XXX, because you specified python3 to be your main interpreter.

You can exit the VE with running "deactivate", (dlbio_example) disappears from your terminal. Note how, when running "python" again, your python version is now python2.7.XXX, the main interpreter outside the VE.

You can enter the VE again via:
```
workon dlbio_example
```

### Install the necessary packages
To install the needed packages run:

```
pip install -r requirements.txt 
```

Now, you should be able to run the code, e.g., "python experiments/model_search/exe_train_model_search.py". Note that, if you don't run the code within the VE, you'll get import errors.
## Important Files

### run_training.py

The starting point of the project. Execute this file to launch the training process.

### train_interfaces.py

Contains classes that define the loss computations and which metrics are saved to the log files.

## Tensorboard

Tensorboard can be used to display training experiments. Infos at https://www.tensorflow.org/tensorboard and [here](https://pytorch.org/docs/stable/tensorboard.html) for using it in pytorch.
Tensor behaves similarly to a jupyter notebook.
In tensorboard, you can visualize model structures, data, and training logs.


Run the exe_train_model_search.py file with the flag CREATE_TBOARD_RUNS set to true to create tensorboard files in the folder 'tboard'.
In the terminal, run 'tensorboard --logdir tboard' to start a tensorboard session.

## ds_natural_images dataset

For the ds_natural_images dataset you'll need to download the dataset on:
https://www.kaggle.com/prasunroy/natural-images
Unzip and remove the data folder (for some reason the data are saved two times).

## Docker

Upon execution of the docker_build.sh script, a Docker-Image will be created. This image
will be based on the Dockerfile. Henceforth, you will always have to rebuild the image
when your dependencies change or are updated (like for example the DLBio-repository) and
you wish to use these updates.
Execute the docker_run.sh script to launch a Docker-Container in your terminal. This
container will be based on the Docker-image that was created by the docker_build.sh.
You can run Files the same way as you would in any other terminal.

More on docker with vs-code:
https://moodle.uni-luebeck.de/mod/forum/discuss.php?d=33479

### docker/docker_build.sh

Script to build a Docker-Image based on a Dockerfile in the same folder. See Section
"Docker" for further information.

### docker/Dockerfile

A file containing all the dependencies and information on the folder structure, etc. of 
the project. This file must be named "Dockerfile"!


### docker_run.sh

Script to launch a Docker-Container based on the image that is generated by the
docker_build.sh script. See Section "Docker" for further information.


### Debugging:

The Debugger has two options. Select "Docker: Python - General" to start debugging the
project from its main file. Select "Docker: Python - Current" to start debugging the
currently in the editor selected file.
More configurations can be added or present configurations can be changed in the
./vscode/launch.json and ./vscode/tasks.json.


## Visdom

Visdom eases to display data remotely. For general infos see their git: [visdom](https://github.com/fossasia/visdom).

Here it is used to display the feature representation of a trained model. 
The represantation is saved in `exe_save_embeddings.py` and plotted in `plot_embeddings.py`.

Visdom can also be used for visualizing images and plots.
Start the visdom server from the command line with `visdom` and add something to it for example like this:
```python

import visdom
from datasets.ds_mnist import get_dataloader

dl = get_dataloader(False, batch_size=16)	
image = dl.dataset.data[0]

vis = visdom.Visdom()
vis.image(image)
``` 

### Installation
You can install the stable version via pip but I would recommend installing it from source to get the current features (See https://github.com/fossasia/visdom#setup).

For plotting the embeddings this has to be done anyway at the moment (march 21). 
Because the `vis.embeddings` uses this package for the t-sne algorithm (https://github.com/lvdmaaten/bhtsne) you will probably have to install this as well. 
For this, clone the repo into visdom/py/visdom/extra_deps and follow the installation steps here: https://github.com/lvdmaaten/bhtsne#installation



