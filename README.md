# ML-RT
ML-RT stands for Machine Learning and Radiative Transfer. It is a project aimed at examining the possibility of using 
machine learning techniques to emulate radiative transfer simulation results.



## Setup

### Software requirements

We assume your system is equipped with the following dependencies:

* Python 3.8 or newer
* bash
* wget
* unzip
* md5sum (optional)

#### System packages
On Debian or Debian-derivatives, e.g. Ubuntu, the required packages should be part of the base installation 
but can be installed using the default package manager if necessary with the following command:
```bash
sudo apt install wget unzip md5sum
```
#### Python modules
Furthermore, the following Python packages are needed:

* pytorch
* numpy (1.20.0 or newer)
* numba (used in the Soft DTW implementation)
* matplotlib
* seaborn
* gdown (to download pre-trained models)

##### pip
The Python dependencies can be installed with `pip` like so:
```bash
pip3 install -r requirements.txt
```

##### conda
In Anaconda (or Miniconda) environments the requirements can be installed like so:
```bash
conda config --add channels conda-forge
conda install --yes --file requirements_conda.txt
```

### Training data download

tba.


## Running Inference
To run pretrained models with custom input, you will need to download the pretrained models first. To do this, 
navigate to the `paper_data` directory via `cd ./paper_data` and run the `pretrained_models.sh` script:
```bash
bash ./pretrained_models.sh
```
This script will download and extract all pretrained models. To run inference on them, navigate to the `src` folder, 
which contains the run scripts, including `inference.py`:
```bash
cd ./src/
python3 inference.py
```
By default, the script will create a plot using the default set parameters in the script. It can easily be adapted to 
run inference with custom parameters. Furthermore, one can select what models to use, measure inference time, and more.
The main function at the bottom of `inference.py` features a sample implementation. 

## Cite this repo / work
tba.

## Feedback
Feedback and questions are welcome. Please get in touch.
