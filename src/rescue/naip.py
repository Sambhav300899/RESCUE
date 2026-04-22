import geopandas as gpd
import planetary_computer
import pystac_client
import rioxarray
from shapely.geometry import box, shape
import warnings

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


def area_of_overlap(items, area_of_interest):

    area_shape = shape(area_of_interest)
    target_area = area_shape.area

    areas = {}
    for item in items:
        overlap_area = shape(item.geometry).intersection(shape(area_of_interest)).area
        areas[item] = overlap_area / target_area

    return areas


def download_naip_for_geojson(geojson_path, time_range, area_overlap_thresh=0.95):
    """
    Download and clip a NAIP tile for the given GeoJSON AOI.

    Returns:
        xarray.DataArray on success, or None when no NAIP imagery matches
        the AOI/time/overlap threshold.
    """
    gdf = gpd.read_file(geojson_path).to_crs(4326)
    polygon = gdf.geometry.union_all()
    if polygon.is_empty:
        warnings.warn(f"Empty geometry in {geojson_path}; returning None")
        return None

    search = catalog.search(
        collections=["naip"], intersects=polygon, datetime=time_range
    )
    items = search.item_collection()
    if len(items) == 0:
        warnings.warn(
            f"No NAIP items for AOI/time_range={time_range} (file={geojson_path}); returning None"
        )
        return None
    area_overlap_dict = area_of_overlap(items, polygon)

    # Print overlap for each item
    print("NAIP item area overlaps for AOI:")
    for item, overlap in area_overlap_dict.items():
        # Print basic info: datetime and overlap
        print(f" - {getattr(item, 'datetime', None)}: {overlap:.3f}")

    area_overlap_dict_filtered = [
        item
        for item, overlap in area_overlap_dict.items()
        if overlap >= area_overlap_thresh
    ]

    if len(area_overlap_dict_filtered) == 0:
        warnings.warn(
            f"No NAIP items meeting overlap>={area_overlap_thresh} for file={geojson_path}; returning None"
        )
        return None

    item_of_interest = max(
        area_overlap_dict_filtered,
        key=lambda x: x.datetime,
    )

    print(f"Selected item datetime: {item_of_interest.datetime}, overlap: {area_overlap_dict[item_of_interest]:.3f}")

    tile = rioxarray.open_rasterio(
        item_of_interest.assets["image"].href, chunks=1024
    ).sel(band=[1, 2, 3, 4])

    reprojected_gdf = gdf.to_crs(tile.rio.crs)
    reprojected_bounds_gdf = get_total_bounds(reprojected_gdf)

    img_clipped = tile.rio.clip(
        reprojected_bounds_gdf.geometry.values,
        reprojected_bounds_gdf.crs,
        drop=True,
    )

    img_clipped = img_clipped.rio.reproject("EPSG:4326")
    img_clipped = img_clipped.rio.clip(
        gdf.geometry.values, 
        gdf.crs, 
        drop = True
    )

    return img_clipped.compute()


def get_total_bounds(gdf):
    gdf = gdf.copy()

    total_bounds = gdf.total_bounds
    minx, miny, maxx, maxy = total_bounds
    bounding_box = box(minx, miny, maxx, maxy)

    return gpd.GeoDataFrame(geometry=[bounding_box], crs=gdf.crs)
