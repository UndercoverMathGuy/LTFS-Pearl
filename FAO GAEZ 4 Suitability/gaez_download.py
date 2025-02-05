import requests
crops = [
    "Alfalfa",
    "Banana",
    "Barley",
    "Biomass sorghum",
    "Buckwheat",
    "Cabbage",
    "Carrot",
    "Cassava",
    "Chickpea",
    "Citrus",
    "Cocoa",
    "Coconut",
    "Coffee",
    "Cotton",
    "Cowpea",
    "Dry pea",
    "Dryland rice",
    "Flax",
    "Foxtail millet",
    "Gram",
    "Groundnut",
    "Jatropha",
    "Maize",
    "Miscanthus",
    "Napier grass",
    "Oat",
    "Oil palm",
    "Olive",
    "Onion",
    "Pearl millet",
    "Phaseolus bean",
    "Pigeonpea",
    "Rapeseed",
    "Reed canary grass",
    "Rubber",
    "Rye",
    "Sorghum",
    "Soybean",
    "Sugarbeet",
    "Sugarcane",
    "Sunflower",
    "Sweet potato",
    "Switchgrass",
    "Tea",
    "Tobacco",
    "Tomato",
    "Wetland rice",
    "Wheat",
    "White potato",
    "Yam"
]
   
for crop in crops:   
    # Define the REST API request URL
    api_url = "https://gaez-services.fao.org/server/rest/services/res05/ImageServer/exportImage"
    params = {
        "bbox": "-179.99999999999997,-89.9999928,179.99998560000003,90.0",
        "bboxSR": "4326",
        "imageSR": "4326",
        "format": "jpgpng",
        "size": "1024,1024",  # Adjust resolution if needed
        "f": "image",
        "crop": f"{crop}",
        "variable":"Output density (potential production divided by total grid cell area)",
        "name": "Suitability and Attainable Yield Symbology",
        "year": "2011-2040",
        "sub_theme_name": "Agro-ecological Attainable Yield",
        "renderer": "Suitability and Attainable Yield Symbology",
    }

    # Make the request
    response = requests.get(api_url, params=params, stream=True)

    # Save image if successful
    if response.status_code == 200:
        with open(f"gaez_raster_{crop}.png", "wb") as file:
            output_dir = "C:/Users/ruhaa/OneDrive/Desktop/Ruhaan/Personal/Projects/LTFS Pearl/LTFS-Pearl/FAO GAEZ 4 Suitability"
            with open(f"{output_dir}/gaez_raster_{crop}.png", "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
        print(f"Raster downloaded: gaez_raster_{crop}.png")
    else:
        print("Failed to fetch image:", response.status_code)
