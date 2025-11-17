import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import sys
import warnings
from rasterio.errors import NotGeoreferencedWarning

# ============================================================================
# CAST 1: NASTAVENI A GEOREFERENCE
# ============================================================================

print(f"\n--- Startuji Cast 1: Georeferencovani obrazku ---")
# id body (pixely a souradnice)
control_points = [
    ((644, 820), (3457000, 5595000)),  
    ((3476, 801), (3463000, 5595000)), 
    ((3502, 3633), (3463000, 5589000)), 
    ((671, 3652), (3457000, 5589000)),
    ((2075, 2227), (3460000, 5592000))  
]

# Vstupni a vystupni soubory
INPUT_RASTER_RAW = 'maska_std_cropped.png'
OUTPUT_GPKG = 'les_final_UTM.gpkg'

# Souradnicove systemy
TARGET_CRS_S42 = 'EPSG:28403'
TARGET_CRS_UTM = 'EPSG:32633'

# ============================================================================
# KROK 1.1: PRIPRAVA TRANSFORMACE A DAT
# ============================================================================

gcps = []
for (pixel_x, pixel_y), (s1942_x, s1942_y) in control_points:
    gcp = GroundControlPoint(row=pixel_y, col=pixel_x, x=s1942_x, y=s1942_y)
    gcps.append(gcp)
print(f"\nPripraveno {len(gcps)} kontrolnich bodu.")

# Vypocitame transformaci z identickych bodu
new_transform_S42 = from_gcps(gcps)
print("Georeferencni transformace (S1942) vypoctena")

# Nacteme surova pixelova data
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    try:
        with rasterio.open(INPUT_RASTER_RAW) as src:
            data = src.read(1)
            src_profile = src.profile # Ulozime si profil (pro rozmery)
            print(f"Zdrojovy soubor '{INPUT_RASTER_RAW}' uspesne nacten.")
    except rasterio.errors.RasterIOError:
        print(f"CHYBA: Soubor '{INPUT_RASTER_RAW}' nenalezen.")
        sys.exit(1)

# --- Konec Casti 1 ---
# Mame v pameti: 'data', 'new_transform_S42', 'TARGET_CRS_S42'

# ============================================================================
# CAST 2: REPROJEKCE (S1942 -> UTM)
# ============================================================================

print(f"\n--- Startuji Cast 2: Reprojekce (Rasterio/GDAL) ---")

# Krok 2.1: Vypocet parametru pro cilovy (UTM) raster
# Zjistime, jak velky bude novy obrazek v UTM
dst_transform, dst_width, dst_height = calculate_default_transform(
    TARGET_CRS_S42,    # Zdroj (S1942)
    TARGET_CRS_UTM,       # Cil (UTM)
    src_profile['width'], # puvodni sirka
    src_profile['height'],# puvodni vyska
    *rasterio.transform.array_bounds(data.shape[0], data.shape[1], new_transform_S42)
)

# Krok 2.2: Priprava prazdneho pole pro reprojektovana data
destination_array = np.empty((dst_height, dst_width), dtype=data.dtype)

# Krok 2.3: Provedeni reprojekce 

reproject(
    source=data,
    destination=destination_array,
    src_transform=new_transform_S42,
    src_crs=TARGET_CRS_S42,
    dst_transform=dst_transform,
    dst_crs=TARGET_CRS_UTM,
    resampling=Resampling.nearest # Zachova 0 a 1
)
print("Reprojekce dokoncena.")

# ============================================================================
# CAST 3: POLYGONIZACE jiz v UTM
# ============================================================================

print(f"\n--- Startuji Cast 3: Polygonizace ---")

polygons = []

# Hledame pixely s hodnotou 1
print("\nHledam pixely s hodnotou 1...")
mask = (destination_array == 1)

# Spustime polygonizaci na datech
results = rasterio.features.shapes(
    destination_array, 
    mask=mask, 
    transform=dst_transform 
)

for geom_dict, value in results:
    polygons.append({
        'geometry': shape(geom_dict), 
        'value': int(value)
    })

print(f"Nalezeno a polygonizovano {len(polygons)} ploch lesa.")

# ============================================================================
# CAST 4: ULOZENI DO GEOPACKAGE
# ============================================================================

if not polygons:
    print("CHYBA: Nebyly nalezeny zadne polygony.")
    sys.exit(1)

try:
    # Vytvorime GeoDataFrame v UTM

    gdf_utm = gpd.GeoDataFrame(
        polygons, 
        geometry='geometry',
        crs=TARGET_CRS_UTM 
    )
    print(f"Vytvoren GeoDataFrame v CRS: {gdf_utm.crs}")
except KeyError:
    print("CHYBA: Selhalo vytvoreni GeoDataFrame.")
    sys.exit(1)

# Ulozime do GeoPackage
gdf_utm.to_file(OUTPUT_GPKG, driver='GPKG')

print(f"\nHotovo! Vse uspesne ulozeno do: {OUTPUT_GPKG}")