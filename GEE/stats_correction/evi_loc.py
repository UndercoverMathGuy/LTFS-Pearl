import pandas as pd

def evi_locs():
    df = pd.read_csv('train_data_cleaned.csv')
    # keep the same index so it lines up with the coords file
    return df[['VILLAGE', 'DISTRICT', 'State', 'Zipcode']]

def return_coordinates_from_file(coord_csv='loc_coords_checkpoint.csv') -> pd.DataFrame:
    """
    Reads the checkpoint CSV you used for geocoding and returns a DataFrame
    of lat/lon indexed exactly like the input df.
    """
    coords = pd.read_csv(coord_csv, index_col=0)
    df = evi_locs()
    coords = coords.reindex(df.index)
    return coords[['lat', 'lon']]

def write_gee_text_file(output_file="ee_features.txt"):
    df = evi_locs()
    coords = return_coordinates_from_file()

    with open(output_file, "w") as f:
        for i, (lat, lon) in coords.iterrows():
            if pd.isna(lat) or pd.isna(lon):
                # skip any that failed to geocode
                continue
            line = f"ee.Feature(ee.Geometry.Point([{lon}, {lat}]), {{id: '{i}_Kharif_2021'}}),"
            f.write(line + "\n")
    print(f"Wrote {output_file}")

if __name__ == '__main__':
    write_gee_text_file()
