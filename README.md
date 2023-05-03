# Geoguessr? I don’t even know her...
This project seeks to (1) create a dataset of geotagged images and (2) build two models: one that is capable of predicting a country that an image comes from and another that predicts the exact lattitude and longitude coordinates of a given image.

## Dataset

---
To build our dataset, we source images from Google's Streetview API. We generate a random coordinate pair on land and check to see if there is a streetview image there. If there is, the image is downloaded, saved to a folder of its country name with its file name being its coordinate pair. The exact routine can be viewed in [google_scraping.py](https://github.com/4D0R/GeoGuessr/blob/a99b0554a2c750334c174ee1fd2ac45b97331636/google_scraping.py) and a visualizaiton of the dataset is in [model_demo.ipynb](https://github.com/4D0R/GeoGuessr/blob/a99b0554a2c750334c174ee1fd2ac45b97331636/model_demo.ipynb). The data itself is available [here](https://www.kaggle.com/datasets/rohanmyer/geotagged-streetview-images-15k).

## Model

---
Both our country classification and coordinate prediction models finetune Resnet-50. The country classification model includes two dropout layers to reduce overfitting. The exact implementations are in [model.py](https://github.com/4D0R/GeoGuessr/blob/a99b0554a2c750334c174ee1fd2ac45b97331636/model.py). The trained models are available on Huggingface as [countryclasssifier](https://huggingface.co/rohanmyer/countryclassifier) and [latlongpredictor](https://huggingface.co/rohanmyer/latlongpredictor).

## Demonstrations

---
For demonstrations of our dataset and models, see [model_demo.ipynb](https://github.com/4D0R/GeoGuessr/blob/a99b0554a2c750334c174ee1fd2ac45b97331636/model_demo.ipynb). This notebook visualizes the locations of each image in our dataset and demonstrates the capabilities and interpretability of our models. 