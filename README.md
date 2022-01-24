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
   git clone https://github.com/80869538/Road_Extraction_Challenge.git
   ```

2. Move to the dictory
   ```sh
   cd ./Road_Extraction_Challenge
   ```
3. Initialize Conda environment
   ```sh
   conda env create -f environment.yml
   ```
4. Activate the environment
   ```sh
   conda activate road_extraction
   ```    
## Data

This dataset contains 6226 satellite images with 50 cm pixel resolution in the RGB format. Each satellite image is paired with a mask image as road labels. The mask is a grayscale image, with white standing for road pixel, and black standing for background. File names for satellite images and the corresponding mask image are "id _sat.jpg" and "id _mask.png". The values in the mask image may not be either 0 or 255. When converting these values to labels, please binarise them via threshold 128.


## Usage

### Dataset

The dataset is available at [Google Drive](https://drive.google.com/file/d/1tB8Jo_wfbz796aTQP8fGnMdTWmujKsqd/view?usp=sharing)

After downloading finished, place Road_Extraction_Dataset folder in the 'data' folder under current work directory.

### Dataset

The dataset is available at [Google Drive](https://drive.google.com/file/d/1tB8Jo_wfbz796aTQP8fGnMdTWmujKsqd/view?usp=sharing)

After downloading finished, place Road_Extraction_Dataset folder in the 'data' folder under current work directory.

We draw the first five input images and their labels with following code:
```
train_features, train_labels = read_RE_images('data/Road_Extraction_Dataset')
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
utils.show_images(imgs, 2, n)
plt.show()
``` 
![The result](assets/images/Sample_Train_Pairs.png)

In the label images, white and black represent borders and background, respectively, while the other colors correspond to different classes. 

## Contact
Andrew Jiang - andrew.jiang81@gmail.com


## Acknowledgement

The dataset used for this project was obtained from the [Road Extraction Challenge Track](https://competitions.codalab.org/competitions/18467#participate-get_starting_kit).  For more details on the dataset refer the related publication - [DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images](https://arxiv.org/abs/1805.06561)

