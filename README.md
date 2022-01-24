# DeepGlobe Road Extraction from Satellite Imagery - A hands-on attempt

<!-- ABOUT THE PROJECT -->
## About The Project

This project uses [DeepGlobe Road Extraction Challenge](https://competitions.codalab.org/competitions/18467#participate-get_starting_kit) as a hands-on practice with image semantic segmentation. 


## Prerequisites

* [Anaconda](https://www.anaconda.com/products/individual)

On MacOS
  ```sh
  brew install --Casks  anaconda
  ```

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Initialize Conda environment
   ```sh
   conda env create -f environment.yml
   ```
3. Activate the environment
   ```sh
   conda activate road_extraction
   ```    
## Data

This dataset contains 6226 satellite images with 50 cm pixel resolution in the RGB format. Each satellite image is paired with a mask image as road labels. The mask is a grayscale image, with white standing for road pixel, and black standing for background. File names for satellite images and the corresponding mask image are "id _sat.jpg" and "id _mask.png". The values in the mask image may not be either 0 or 255. When converting these values to labels, please binarise them via threshold 128.

**Download Training Data Set**
The dataset is available at [Google Drive](https://drive.google.com/file/d/1tB8Jo_wfbz796aTQP8fGnMdTWmujKsqd/view?usp=sharing)
## Contact
Andrew Jiang - andrew.jiang81@gmail.com


## Acknowledgement

The dataset used for this project was obtained from the [Road Extraction Challenge Track](https://competitions.codalab.org/competitions/18467#participate-get_starting_kit).  For more details on the dataset refer the related publication - [DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images](https://arxiv.org/abs/1805.06561)

