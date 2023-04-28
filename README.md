# GeoGuessr

To load in data:
`gsutil -m cp -n -r gs://geoguessr-imgs/streetviews data/`

To run the country classificaiton model:
`python simple_model.py`

To run the lat-long prediction model:
`python simple_model.py --lat_long`
