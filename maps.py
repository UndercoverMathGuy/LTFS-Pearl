import requests as r
import json
import pandas as pd
import urllib.parse as parse
from constants import google_maps_api_key

def request_maps(village:str, district: str, state:str, zip:str):
    address = f"{village}+{district}+{state}+{zip}"
    address_url = parse.quote_plus(address)
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={google_maps_api_key}"
    response = r.get(url)
    data = json.loads(response.text)
    lat = data["results"][0]["geometry"]["location"]["lat"]
    lon = data["results"][0]["geometry"]["location"]["lng"]
    latlon = (lat, lon)
    return latlon
