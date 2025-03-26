import requests as r
import json

def get_soilgrids(lat, lon):
    url = f'https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lon}&lat={lat}&number_class=5'
    response = r.get(url)
    data = json.loads(response.text)
    return data['wrb_class_name']

lat = 52.379189
lon = 17.00326

print(get_soilgrids(lat, lon))