import requests
import threading
from argparse import ArgumentParser
from tqdm.auto import tqdm
import random

from faker import Faker

from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.get_bucket("geoguessr-imgs")

class StreetViewer(object):
    def __init__(self, api_key, location, country, size="640x640",
                 image_folder='streetviews/', meta_folder="./meta/"):
        """
        This class handles a single API request to the Google Static Street View API
        api_key: obtain it from your Google Cloud Platform console
        location: the address string or a (lat, lng) tuple
        size: returned picture size. maximum is 640*640
        image_folder: directory to save the returned image from request
        meta_folder: directory to save the returned metadata from request
        verbose: whether to print the processing status of the request
        """
        # input params are saved as attributes for later reference
        self._key = api_key
        self.location = location
        self.country = country
        self.size = size
        self.image_folder = image_folder
        self.meta_folder = meta_folder
        # call parames are saved as internal params
        self._meta_params = dict(key=self._key,
                                location=self.location)
        self._pic_params = dict(key=self._key,
                               location=self.location,
                               size=self.size)
    
    def get_meta(self):
        """
        Method to query the metadata of the address
        """
        # saving the metadata as json for later usage
        # "/"s are removed to avoid confusion on directory
        self.meta_path = "{}meta_{}.json".format(
            self.meta_folder, self.location.replace("/", ""))
        self._meta_response = requests.get(
            'https://maps.googleapis.com/maps/api/streetview/metadata?',
            params=self._meta_params)
        # turning the contents as meta_info attribute
        self.meta_info = self._meta_response.json()
        # meta_status attribute is used in get_pic method to avoid
        # query when no picture will be available
        self.meta_status = self.meta_info['status']
        if self._meta_response.ok:
            # with open(self.meta_path, 'w') as file:
            #     json.dump(self.meta_info, file)
            self._meta_response.close()
            return True
        else:
            self._meta_response.close()
            return False
    
    def get_pic(self):
        """
        Method to query the StreetView picture and save to local directory
        """
        if not self.get_meta():
            return False
        # define path to save picture and headers
        self.pic_path = f"{self.image_folder}{self.country}/{self.location.replace('/', '')}.jpg"
        self.header_path = f"{self.meta_folder}header_{self.location.replace('/', '')}.json"
        
        # only when meta_status is OK will the code run to query picture (cost incurred)
        if self.meta_status == 'OK':
            self._pic_response = requests.get(
                'https://maps.googleapis.com/maps/api/streetview?',
                params=self._pic_params)
            # self.pic_header = dict(self._pic_response.headers)
            if self._pic_response.ok:
                blob = bucket.blob(self.pic_path)
                blob.content_type = "image/jpeg"
                with blob.open("wb") as f:
                    f.write(self._pic_response.content)
                # with open(self.pic_path, 'wb') as file:
                #     file.write(self._pic_response.content)
                # with open(self.header_path, 'w') as file:
                #     json.dump(self.pic_header, file)
                self._pic_response.close()
                return True
            else:
                self._pic_response.close()
                return False

# def generate_coordinate():
#     return fake.location_on_land() # (latitude, longitude, place name, two-letter country code, timezone)

def get_n_images(n, index):
    fake = Faker()
    Faker.seed(random.randint(0, 100000))

    i = 0
    already_downloaded = 0
    no_image = 0
    bad_country = 0

    pbar = tqdm(total=n, bar_format='{desc:20}{percentage:3.0f}%|{bar:15}{r_bar}')
    while i < n:
        pbar.set_description(f"In Bucket: {already_downloaded}, No Image: {no_image}, Bad Country: {bad_country}\t")
        latlong = fake.location_on_land() # (latitude, longitude, place name, two-letter country code, timezone)
        if latlong[3] in ["US", "GB", "FR", "JP", "IT", "BE", "RU", "NL", "DE", "IN", "ES", "ID", "AU", "TH", "BR", "MX", "CA", "PH"]:
            bad_country += 1
            continue
        lat = str(float(latlong[0]) + round(random.uniform(-0.5, 0.5), 5))
        long = str(float(latlong[1]) + round(random.uniform(-0.5, 0.5), 5))
        blobs = storage_client.list_blobs("geoguessr-imgs", prefix=f"streetviews/{latlong[3]}",)
        for blob in blobs:
            if blob.name == f"streetviews/{latlong[3]}/{lat},{long}.jpg":
                already_downloaded += 1
                break
        else:
            api_key = "AIzaSyDPKuAFZQk76T4eSehLw4Qs3eJ5jfyCYx4"
            gwu_viewer = StreetViewer(api_key=api_key,
                                    location= lat + "," + long,
                                    country=latlong[3])
            # gwu_viewer.get_meta()
            if gwu_viewer.get_pic():
                i += 1
                pbar.update(1)
            else:
                no_image += 1

def main(total_images):
    threads = list()
    NUM_THREADS = 20
    for index in range(NUM_THREADS):
        x = threading.Thread(target=get_n_images, args=(total_images // NUM_THREADS, index,))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    args = parser.parse_args()
    main(args.n)