# Denoising Diffusion Probabilistic Model (DDPM) Reimplementation ✨

This repository contains a reimplementation of the Denoising Diffusion Probabilistic Model (DDPM) for image generation, based on the [seminal paper by Ho et al.](https://arxiv.org/pdf/2006.11239), with elements from other reference literature on denoising diffusion models, such as [Improved denoising diffusion probabilistic models](https://arxiv.org/pdf/2102.09672) by Nichol and Dhariwal, and [Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models](https://arxiv.org/pdf/2304.12526) by Wang et al. The project includes scripts for training the diffusion model and for generating new images using the trained model.

## Table of Contents 📚

- [Denoising Diffusion Probabilistic Model (DDPM) Reimplementation ✨](#denoising-diffusion-probabilistic-model-ddpm-reimplementation-)
  - [Table of Contents 📚](#table-of-contents-)
  - [Overview 🌟](#overview-)
  - [Installation 🛠️](#installation-️)
  - [Usage 🚀](#usage-)
    - [Configuration ⚙️](#configuration-️)
    - [Training 🎓](#training-)
    - [Sampling 🎨](#sampling-)
  - [Results 📈](#results-)
  - [Contributing 🤝](#contributing-)
  - [License 📄](#license-)

## Overview 🌟

This project reimplements the Denoising Diffusion Probabilistic Model as described in the seminal paper. The main scripts included are:

- `train.py`: This script trains the diffusion model using the configurations specified in `config.yaml`.
- `sample.py`: This script generates new images from the trained diffusion model using the settings defined in `config.yaml`.

## Installation 🛠️

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/danilodjor/denoising-diffusion
cd denoising-diffusion
python -m venv venv/denoising_diffusion
source venv/denoising_diffusion/bin/activate
pip install -r requirements.txt
```

## Usage 🚀

The `utils` folder contains essential scripts and files for defining the neural network model used in the diffusion process. It includes the implementation of the time positional encoding, the scheduler class definition, and functions for configuring and transforming the training data. The primary scripts for training and sampling the diffusion model are `train.py` and `sample.py`, respectively.


### Configuration ⚙️
All configurations for training and sampling are managed through the `config.yaml` file. Below is an overview of the key settings:
- Diffusion Parameters:
  - num_steps: Number of denoising steps during sampling.
  - scheduler: Type of noise scheduler to use. One of ['linear', 'cosine'].
- Training Parameters:
  - num_epochs: Number of training epochs.
  - learning_rate: Learning rate for the optimizer.
  - batch_size: Number of samples per batch.
- Data Patameters:
  - dataset: Name of the huggingface image dataset that will be used for training.
  - img_size: Size of the square images that will be sampled.
- Logging Parameters:
  - log_dir: Path to the directory in which tensorboard log files will be saved.
  - save_dir: Path to the directory in which model weights will be saved.
- Sampling Parameters:
  - num_sample_imgs: String in format `NxM` where `N` is the number of rows and `M` is the number of columns of the generated grid of images.
  - model_path: Path to the trained model weights used for sampling.
  - save_dir: Path to the directory in which generated images will be saved.
    

Example configuration (`config.yaml`):
```
diffusion:
  num_steps: 200
  scheduler: linear

training:
  batch_size: 128
  learning_rate: 0.0002
  num_epochs: 30

data:
  dataset: mnist
  img_size: 64

logging:
  log_dir: runs
  save_dir: models

sampling:
  num_sample_imgs: 5x5
  save_dir: generated
  model_path: models\2024_05_27_10_00\best_model_ep1.pth
```


### Training 🎓
To train the diffusion model, simply run:

`python train.py`

All training parameters such as the dataset, number of epochs, learning rate, batch size, number of denoising steps, type of noise scheduler are specified in the `config.yaml` file.
Each training log file is saved in its own subdirectory with format "YYYY_MM_DD_HH_MM". Same holds for model weights. The format denotes the time at which training was initiated.

### Sampling 🎨
To generate new images using the trained model, run:

`python sample.py`

The sampling process, including the selection of trained model weights, number of denoising steps, and type of noise scheduler, is configured in `config.yaml`.

## Results 📈
You can find example generated images in the generated directory within the project directory after running the sample.py script. All generated images are named in the format "generated_YYYY_MM_DD_HH_MM.png".

The plot below illustrates the difference in $\hat{\alpha}_t$ values at various time points for linear and cosine noise schedules:

<img src="images\alpha_hat_t.png" alt="alpha_t plot" width="400"/>

Using the cosine schedule, the following noising effect is achieved:

<img src="images\noising.png" alt="noising effect" width="800"/>

Training on the huggan/smithsonian_butterflies_subset dataset for approximately 5000 steps with a batch size of 128, a linear noise scheduler, and a learning rate of 2e-4 has produced the following loss curve:

<img src="images\training-curve2.png" alt="training curve" width="400"/>

Finally, here are some examples of butterflies generated by the diffusion model at different stages of training (progressing from left to right):

<img src="images\generated_large.png" alt="generated butterflies" width="800"/>


## Contributing 🤝
Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue.

## License 📄
This project is licensed under the MIT License. See the `LICENSE.md` file for details.