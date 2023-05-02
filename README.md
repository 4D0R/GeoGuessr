# GeoGuessr

To load in data:
`gsutil -m cp -n -r gs://geoguessr-imgs/streetviews data/`

To run the country classificaiton model:
`python model.py`

To run the lat-long prediction model:
`python model.py --lat_long`
