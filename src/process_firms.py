"""Filter NASA FIRMS MODIS (C6.1, standard processing) country-year files to fires
relevant for Bangkok air quality.

Input : raw country-year CSVs from https://firms.modaps.eosdis.nasa.gov/data/country/modis/<YEAR>/modis_<YEAR>_<COUNTRY>.csv
Output: one CSV of vegetation-fire detections (type == 0) with confidence >= 30
        within MAX_KM of central Bangkok, plus distance and country columns.

Usage: python src/process_firms.py <raw_dir> <out_csv>
"""
import sys, glob, os
import numpy as np, pandas as pd

BKK_LAT, BKK_LON = 13.7563, 100.5018   # central Bangkok
MAX_KM = 600
MIN_CONF = 30                           # drop MODIS "low confidence" (<30) detections

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

def main(raw_dir, out_csv):
    parts, log = [], []
    for f in sorted(glob.glob(os.path.join(raw_dir, 'modis_*_*.csv'))):
        country = os.path.basename(f)[:-4].split('_', 2)[2]
        d = pd.read_csv(f)
        n0 = len(d)
        d = d[(d['type'] == 0) & (d['confidence'] >= MIN_CONF)].copy()
        d['dist_km'] = haversine_km(BKK_LAT, BKK_LON, d['latitude'], d['longitude'])
        d = d[d['dist_km'] <= MAX_KM]
        d['country'] = country
        parts.append(d[['acq_date', 'acq_time', 'latitude', 'longitude', 'confidence', 'frp',
                        'daynight', 'satellite', 'country', 'dist_km']])
        log.append((os.path.basename(f), n0, len(d)))
    out = pd.concat(parts, ignore_index=True).sort_values(['acq_date', 'acq_time'])
    out['dist_km'] = out['dist_km'].round(1)
    out.to_csv(out_csv, index=False)
    for row in log:
        print(*row)
    print('kept', len(out))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
