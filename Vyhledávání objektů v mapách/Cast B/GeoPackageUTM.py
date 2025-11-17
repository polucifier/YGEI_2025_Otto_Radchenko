import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps
import sys

# ============================================================================
# ČÁST 1: NASTAVENÍ A GEOREFERENCE
# ============================================================================
print(f"\n--- Startuji Část 1: Georeferencování obrázku ---")
# kontrolni body (pixely a EPSG:28403)
control_points = [
    ((644, 820), (3457000, 5595000)),  
    ((3476, 801), (3463000, 5595000)), 
    ((3502, 3633), (3463000, 5589000)), 
    ((671, 3652), (3457000, 5589000)),
    ((2075, 2227), (3460000, 5592000))  
]

# Vstupní a výstupní soubory
INPUT_RASTER_RAW = 'maska_std_cropped.png'
OUTPUT_GPKG = 'les_final_UTM.gpkg'

# Souřadnicové systémy
TARGET_CRS_KROVAK = 'EPSG:28403'
TARGET_CRS_UTM = 'EPSG:32633'

# ============================================================================
# KROK 1.1: PŘEVOD BODŮ A VÝPOČET TRANSFORMACE
# ============================================================================

gcps = []
for (pixel_x, pixel_y), (s1942_x, s1942_y) in control_points:
    gcp = GroundControlPoint(row=pixel_y, col=pixel_x, x=s1942_x, y=s1942_y)
    gcps.append(gcp)

print(f"\nPřipraveno {len(gcps)} kontrolních bodů.")

# Vypočítáme novou transformaci, kterou budeme potřebovat v Části 2

new_transform = from_gcps(gcps)
print("Georeferenční transformace vypočtena")

# ============================================================================
# KROK 1.2: NAČTENÍ SUROVÝCH PIXELOVÝCH DAT
# ============================================================================

try:
    with rasterio.open(INPUT_RASTER_RAW) as src:
        # Načteme pixelová data (hodnoty 0 a 1) do paměti
        data = src.read(1)
        print(f"Zdrojový soubor '{INPUT_RASTER_RAW}' úspěšně načten.")
except rasterio.errors.RasterioIOError:
    print(f"CHYBA: Soubor '{INPUT_RASTER_RAW}' nenalezen.")
    sys.exit(1)
except Exception as e:
    print(f"Nastal jiný problém při čtení souboru: {e}")
    sys.exit(1)

# --- Konec Části 1 ---
# Nyní máme v paměti vše, co potřebujeme:
# 1. 'data' (numpy pole s pixely 0 a 1)
# 2. 'new_transform' (objekt transformace)
# 3. 'TARGET_CRS_KROVAK' (náš CRS)

# ============================================================================
# ČÁST 2: POLYGONIZACE A PŘEVOD DO UTM
# ============================================================================

print(f"\n--- Startuji Část 2: Polygonizace a převod do UTM ---")

polygons = []

# ============================================================================
# KROK 2.1: POLYGONIZACE (RASTR -> VEKTOR)
# ============================================================================

# Hledáme pixely s hodnotou 1
print("\nHledám pixely s hodnotou 1...")
mask = (data == 1)

# Spustíme polygonizaci

results = rasterio.features.shapes(
    data, 
    mask=mask, 
    transform=new_transform 
)

for geom_dict, value in results:
    polygons.append({
        'geometry': shape(geom_dict), 
        'value': int(value)
    })

print(f"Nalezeno a polygonizováno {len(polygons)} ploch lesa.")

# ============================================================================
# KROK 2.2: VYTVOŘENÍ GEODATAFRAME A KONTROLA
# ============================================================================

if not polygons:
    print("CHYBA: Nebyly nalezeny žádné polygony.")
    sys.exit(1)

try:
    # Vytvoříme GeoDataFrame
    # POUŽIJEME 'TARGET_CRS_KROVAK' jako náš CRS
    gdf_krovak = gpd.GeoDataFrame(
        polygons, 
        geometry='geometry',
        crs=TARGET_CRS_KROVAK 
    )
    print(f"Vytvořen GeoDataFrame v CRS: {gdf_krovak.crs}")
except KeyError:
    print("CHYBA: Selhalo vytvoření GeoDataFrame.")
    sys.exit(1)

# ============================================================================
# KROK 2.3: PŘEPROJEKCE DO UTM
# ============================================================================

print(f"Převádím do cílového CRS: {TARGET_CRS_UTM} (UTM 33N)...")
gdf_utm = gdf_krovak.to_crs(TARGET_CRS_UTM)

# ============================================================================
# KROK 2.4: ULOŽENÍ DO GEOPACKAGE
# ============================================================================

gdf_utm.to_file(OUTPUT_GPKG, driver='GPKG')

print(f"\nHotovo! Vše úspěšně uloženo do: {OUTPUT_GPKG}")