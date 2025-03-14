import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from shapely.geometry import Point, shape
from shapely.validation import make_valid
from shapely.wkt import dumps
from typing import Optional
import numpy as np
import folium
from streamlit_folium import folium_static
from models import Address, Coordinates, Property, SlopeData, EnvironmentalCheck
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Nominatim geocoder
geolocator = Nominatim(user_agent="geotech_mvp_app")

# Paths to GeoJSON files
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
    """Convert an address to latitude/longitude coordinates using Nominatim."""
    try:
        location = geolocator.geocode(address.full_address())
        if location:
            return Coordinates(latitude=location.latitude, longitude=location.longitude)
        else:
            raise ValueError("Address not found in Mercer Island, WA.")
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logging.error(f"Geocoding error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during geocoding: {e}")
        return None

def load_geojson(file_path: str) -> gpd.GeoDataFrame:
    """Load a GeoJSON file into a GeoDataFrame."""
    try:
        return gpd.read_file(file_path)
    except Exception as e:
        logging.error(f"Error loading GeoJSON file {file_path}: {e}")
        raise

def extract_property(coordinates: Coordinates) -> Optional[Property]:
    """Extract property data based on geocoded coordinates."""
    try:
        properties_gdf = load_geojson(PROPERTY_FILE)
        point = Point(coordinates.longitude, coordinates.latitude)
        for idx, row in properties_gdf.iterrows():
            if row.geometry.contains(point):
                return Property(parcel_id=row.get("PARCEL_ID", "unknown"), geometry=row.geometry.__geo_interface__)
        logging.warning("No property found at the given coordinates.")
        return None
    except Exception as e:
        logging.error(f"Error extracting property: {e}")
        return None

def calculate_slope(property: Property) -> Optional[SlopeData]:
    """Calculate slope data by intersecting property with contour lines."""
    try:
        contours_gdf = load_geojson(CONTOUR_FILE).to_crs("EPSG:32610")
        logging.debug(f"Loaded {len(contours_gdf)} contours with columns: {contours_gdf.columns.tolist()}")
        
        property_geom = shape(property.geometry)
        property_geom_proj = gpd.GeoSeries([property_geom], crs="EPSG:4326").to_crs("EPSG:32610")[0]
        
        intersections = contours_gdf[contours_gdf.intersects(property_geom_proj)].copy()
        intersections['geometry'] = intersections.geometry.intersection(property_geom_proj)
        intersections = intersections[~intersections.geometry.is_empty].sort_values(by="Elevation")
        logging.debug(f"Found {len(intersections)} intersections")
        
        if len(intersections) < 2:
            logging.warning("Insufficient contour intersections found for the property.")
            return SlopeData(average_slope=0.0, max_slope=0.0)
        
        if "Elevation" not in intersections.columns:
            logging.warning("Elevation column not found in contour data. Using default elevation.")
            elevations = np.array([0.0])
        else:
            elevations = intersections["Elevation"].values
        logging.debug(f"Elevations: {elevations}")
        
        slopes = []
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
                logging.debug(f"Pair {i}: elev_diff={elev_diff}, dist={dist}, slope={slope_deg}")
        
        if not slopes:
            logging.warning("No valid slopes calculated after filtering.")
            return SlopeData(average_slope=0.0, max_slope=0.0)
        
        avg_slope = np.mean(slopes)
        max_slope = np.max(slopes)
        logging.debug(f"Calculated slopes: {slopes}, Average: {avg_slope}, Max: {max_slope}")
        
        # Sanity check for extreme slopes
        if avg_slope > 45:  # >100% slope equivalent
            logging.warning(f"Extreme slope detected: {avg_slope} degrees. Verify contour data accuracy.")
        
        return SlopeData(average_slope=avg_slope, max_slope=max_slope)
    except Exception as e:
        logging.error(f"Error calculating slope: {e}")
        return SlopeData(average_slope=0.0, max_slope=0.0)

def check_environmental_hazards(property: Property) -> Optional[EnvironmentalCheck]:
    """Check if the property intersects environmental hazard layers."""
    try:
        property_geom = shape(property.geometry)
        hazards = {}
        for hazard_type, file_path in HAZARD_FILES.items():
            hazard_gdf = load_geojson(file_path)
            intersects = any(hazard_gdf.intersects(property_geom))
            hazards[hazard_type] = intersects
        return EnvironmentalCheck(
            erosion=hazards["erosion"],
            potential_slide=hazards["potential_slide"],
            seismic=hazards["seismic"],
            steep_slope=hazards["steep_slope"],
            watercourse=hazards["watercourse"],
        )
    except Exception as e:
        logging.error(f"Error checking environmental hazards: {e}")
        return None

def create_map(coordinates: Coordinates, property: Property, geojson_files: dict) -> None:
    """Create an interactive map with all GeoJSON layers using Folium."""
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
            logging.debug("Property geometry is invalid, attempting to fix...")
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
        logging.error(f"Error creating map: {e}")