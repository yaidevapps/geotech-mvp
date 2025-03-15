import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from shapely.geometry import Point, shape
from shapely.validation import make_valid
from typing import Optional
import numpy as np
import folium
from streamlit_folium import folium_static
from models import Address, Coordinates, Property, SlopeData, EnvironmentalCheck
import logging

logging.basicConfig(level=logging.DEBUG)
geolocator = Nominatim(user_agent="geotech_mvp_app")

PROPERTY_FILE = "data/Mercer_Island_Basemap_Data_Layers_PropertyLine.geojson"
CONTOUR_FILE = "data/Mercer_Island_Environmental_Layers_10ftLidarContours.geojson"
HAZARD_FILES = {
    "erosion": "data/Mercer_Island_Environmental_Layers_Erosion.geojson",
    "potential_slide": "data/Mercer_Island_Environmental_Layers_PotentialSlideAreas.geojson",
    "seismic": "data/Mercer_Island_Environmental_Layers_Seismic.geojson",
    "steep_slope": "data/Mercer_Island_Environmental_Layers_SteepSlope.geojson",
    "watercourse": "data/Mercer_Island_Environmental_Layers_WatercourseBufferSetback.geojson",
}

def geocode_address(address: Address) -> Optional[Coordinates]:
    try:
        location = geolocator.geocode(address.full_address())
        if location:
            return Coordinates(latitude=location.latitude, longitude=location.longitude)
        else:
            raise ValueError("Address not found in Mercer Island, WA.")
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logging.error(f"Geocoding error for {address.full_address()}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during geocoding of {address.full_address()}: {e}")
        return None

def load_geojson(file_path: str) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(file_path)
    except Exception as e:
        logging.error(f"Error loading GeoJSON file {file_path}: {e}")
        raise

def extract_property(coordinates: Coordinates) -> Optional[Property]:
    try:
        properties_gdf = load_geojson(PROPERTY_FILE)
        if "PIN" not in properties_gdf.columns:
            logging.error(f"PIN missing in {PROPERTY_FILE}. Available columns: {properties_gdf.columns}")
            raise ValueError("PIN field not found in property GeoJSON")
        point = Point(coordinates.longitude, coordinates.latitude)
        for idx, row in properties_gdf.iterrows():
            if row.geometry.contains(point):
                return Property(parcel_id=row["PIN"], geometry=row.geometry.__geo_interface__)
        logging.warning(f"No property found at coordinates ({coordinates.latitude}, {coordinates.longitude}).")
        return None
    except Exception as e:
        logging.error(f"Error extracting property at ({coordinates.latitude}, {coordinates.longitude}): {e}")
        return None

def calculate_slope(property: Property) -> Optional[SlopeData]:
    try:
        contours_gdf = load_geojson(CONTOUR_FILE).to_crs("EPSG:32610")
        logging.debug(f"Loaded {len(contours_gdf)} contours with columns: {contours_gdf.columns.tolist()}")
        
        property_geom = shape(property.geometry)
        property_geom_proj = gpd.GeoSeries([property_geom], crs="EPSG:4326").to_crs("EPSG:32610")[0]
        
        intersections = contours_gdf[contours_gdf.intersects(property_geom_proj)].copy()
        if len(intersections) < 2:
            logging.warning(f"Only {len(intersections)} intersections for parcel {property.parcel_id}. Applying 10m buffer...")
            buffered_geom = property_geom_proj.buffer(10)
            intersections = contours_gdf[contours_gdf.intersects(buffered_geom)].copy()
        
        intersections['geometry'] = intersections.geometry.intersection(property_geom_proj)
        intersections = intersections[~intersections.geometry.is_empty].sort_values(by="Elevation")
        logging.debug(f"Found {len(intersections)} contour intersections for parcel {property.parcel_id}")
        
        if len(intersections) < 2:
            logging.warning(f"Insufficient contour intersections ({len(intersections)}) for parcel {property.parcel_id}. Assuming flat slope.")
            return SlopeData(average_slope=0.0, max_slope=0.0)
        
        if "Elevation" not in intersections.columns:
            logging.warning("Elevation column not found in contour data. Using default elevation.")
            elevations = np.array([0.0])
        else:
            elevations = intersections["Elevation"].values
        logging.debug(f"Elevations: {elevations}")
        
        slopes = []
        distances = []
        for i in range(len(intersections) - 1):
            elev_diff = abs(elevations[i + 1] - elevations[i])
            geom1 = intersections.iloc[i].geometry
            geom2 = intersections.iloc[i + 1].geometry
            centroid1 = geom1.centroid
            centroid2 = geom2.centroid
            dist = centroid1.distance(centroid2)
            if 0.5 < dist <= 1000:
                slope_deg = np.degrees(np.arctan(elev_diff / dist))
                slopes.append(slope_deg)
                distances.append(dist)
                logging.debug(f"Pair {i}: elev_diff={elev_diff}m, dist={dist}m, slope={slope_deg}°")
        
        if not slopes:
            logging.warning(f"No valid slopes calculated for parcel {property.parcel_id} after filtering. Assuming flat slope.")
            return SlopeData(average_slope=0.0, max_slope=0.0)
        
        avg_slope = np.mean(slopes)
        max_slope = np.max(slopes)
        avg_distance = np.mean(distances)
        logging.info(f"Calculated slopes for parcel {property.parcel_id}: {slopes}, Average: {avg_slope}°, Max: {max_slope}°, Avg Distance: {avg_distance}m")
        
        if avg_slope > 45:
            logging.warning(f"Extreme slope detected: {avg_slope}° for parcel {property.parcel_id}. Verify contour data accuracy.")
        
        erosion_gdf = load_geojson(HAZARD_FILES["erosion"]).to_crs("EPSG:32610")
        property_buffer = property_geom_proj.buffer(100)
        lake_proximity = erosion_gdf.intersects(property_buffer).any()
        if lake_proximity and avg_slope < 5:
            logging.warning(
                f"Parcel {property.parcel_id} near lake but slope is flat ({avg_slope}°). "
                "Possible contour data omission—recommend topographic survey."
            )
            if max_slope < 15:
                logging.info(f"Adjusting max_slope to 15° for lakefront context on parcel {property.parcel_id}.")
                max_slope = 15.0
        
        return SlopeData(average_slope=avg_slope, max_slope=max_slope, average_distance=avg_distance)
    except Exception as e:
        logging.error(f"Error calculating slope for parcel {property.parcel_id}: {e}")
        return SlopeData(average_slope=0.0, max_slope=0.0)

def check_environmental_hazards(property: Property) -> Optional[EnvironmentalCheck]:
    try:
        property_geom = shape(property.geometry)
        property_gdf = gpd.GeoSeries([property_geom], crs="EPSG:4326").to_crs("EPSG:32610")
        property_geom_proj = property_gdf[0]
        property_area = property_geom_proj.area
        
        hazards = {}
        for hazard_type, file_path in HAZARD_FILES.items():
            hazard_gdf = load_geojson(file_path).to_crs("EPSG:32610")
            intersection_area = hazard_gdf.intersection(property_geom_proj).area.sum()
            overlap_ratio = intersection_area / property_area if property_area > 0 else 0
            intersects = overlap_ratio > 0.1  # Significant overlap threshold
            hazards[hazard_type] = intersects
            logging.debug(f"Checked {hazard_type} for parcel {property.parcel_id}: "
                         f"{'Present' if intersects else 'Not Present'} (Overlap: {overlap_ratio:.2%})")
        return EnvironmentalCheck(
            erosion=hazards["erosion"],
            potential_slide=hazards["potential_slide"],
            seismic=hazards["seismic"],
            steep_slope=hazards["steep_slope"],
            watercourse=hazards["watercourse"],
        )
    except Exception as e:
        logging.error(f"Error checking environmental hazards for parcel {property.parcel_id}: {e}")
        return None

def create_map(coordinates: Coordinates, property: Property, geojson_files: dict) -> None:
    try:
        m = folium.Map(location=[coordinates.latitude, coordinates.longitude], zoom_start=15)
        
        layer_styles = {
            "Property Lines": {"color": "lightgrey", "weight": .5, "fill": False},
            "Contours": {"color": "gray", "weight": .75, "fill": False},
            "Erosion Hazard": {"fillColor": "yellow", "color": "yellow", "weight": 1, "fillOpacity": 0.3},
            "Potential Slide Hazard": {"fillColor": "blue", "color": "blue", "weight": 1, "fillOpacity": 0.2},
            "Seismic Hazard": {"fillColor": "pink", "color": "pink", "weight": 1, "fillOpacity": 0.3},
            "Steep Slope Hazard": {"fillColor": "orange", "color": "orange", "weight": 1, "fillOpacity": 0.3},
            "Watercourse Buffer": {"fillColor": "lightblue", "color": "lightblue", "weight": 1, "fillOpacity": 0.4},
        }

        for layer_name, file_path in geojson_files.items():
            gdf = load_geojson(file_path).set_crs("EPSG:4326", allow_override=True)
            style = layer_styles.get(layer_name, {"fillColor": "gray", "color": "black", "weight": 1, "fillOpacity": 0.3})
            folium.GeoJson(
                gdf,
                name=layer_name.capitalize(),
                style_function=lambda x, s=style: s,
                show=True
            ).add_to(m)
        
        property_geom = shape(property.geometry)
        if not property_geom.is_valid:
            logging.debug(f"Property geometry for parcel {property.parcel_id} is invalid, attempting to fix...")
            property_geom = make_valid(property_geom)
        property_gdf = gpd.GeoDataFrame([property_geom], columns=["geometry"], crs="EPSG:4326")
        folium.GeoJson(
            property_gdf,
            name="Property",
            style_function=lambda x: {
                "fillColor": "green",
                "color": "blue",
                "weight": 2,
                "fillOpacity": 0.7,
                "fill": True
            },
            show=True
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        folium_static(m)
    except Exception as e:
        logging.error(f"Error creating map for parcel {property.parcel_id}: {e}")