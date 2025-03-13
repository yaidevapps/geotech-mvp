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

# Initialize Nominatim geocoder with a custom user agent
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
    """
    Convert an address to latitude/longitude coordinates using Nominatim.
    
    Args:
        address: Address model instance containing address details.
    
    Returns:
        Coordinates model instance with latitude and longitude, or None if geocoding fails.
    """
    try:
        location = geolocator.geocode(address.full_address())
        if location:
            return Coordinates(latitude=location.latitude, longitude=location.longitude)
        else:
            raise ValueError("Address not found in Mercer Island, WA.")
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during geocoding: {e}")
        return None

def load_geojson(file_path: str) -> gpd.GeoDataFrame:
    """
    Load a GeoJSON file into a GeoDataFrame.
    
    Args:
        file_path: Path to the GeoJSON file.
    
    Returns:
        GeoDataFrame containing the GeoJSON data.
    
    Raises:
        Exception: If the file cannot be loaded.
    """
    try:
        return gpd.read_file(file_path)
    except Exception as e:
        print(f"Error loading GeoJSON file {file_path}: {e}")
        raise

def extract_property(coordinates: Coordinates) -> Optional[Property]:
    """
    Extract property data based on geocoded coordinates.
    
    Args:
        coordinates: Coordinates model instance with latitude and longitude.
    
    Returns:
        Property model instance with parcel ID and geometry, or None if no property is found.
    """
    try:
        properties_gdf = load_geojson(PROPERTY_FILE)
        point = Point(coordinates.longitude, coordinates.latitude)
        # Find the property polygon that contains the point
        for idx, row in properties_gdf.iterrows():
            if row.geometry.contains(point):
                return Property(parcel_id=row.get("PARCEL_ID", "unknown"), geometry=row.geometry.__geo_interface__)
        print("No property found at the given coordinates.")
        return None
    except Exception as e:
        print(f"Error extracting property: {e}")
        return None

def calculate_slope(property: Property) -> Optional[SlopeData]:
    """
    Calculate slope data by intersecting property with contour lines.
    
    Args:
        property: Property model instance with geometry.
    
    Returns:
        SlopeData model instance with average and max slope, or default (0, 0) if calculation fails.
    """
    try:
        # Load contours and project to UTM Zone 10N (meters) for accurate distance
        contours_gdf = load_geojson(CONTOUR_FILE).to_crs("EPSG:32610")
        print(f"Loaded {len(contours_gdf)} contours with columns: {contours_gdf.columns.tolist()}")
        
        # Convert property geometry to Shapely object and project
        property_geom = shape(property.geometry)
        property_geom_proj = gpd.GeoSeries([property_geom], crs="EPSG:4326").to_crs("EPSG:32610")[0]
        
        # Find intersections and clip contours to property geometry
        intersections = contours_gdf[contours_gdf.intersects(property_geom_proj)].copy()
        intersections['geometry'] = intersections.geometry.intersection(property_geom_proj)
        intersections = intersections[~intersections.geometry.is_empty].sort_values(by="Elevation")
        print(f"Found {len(intersections)} intersections")
        
        if len(intersections) < 2:
            print("Insufficient contour intersections found for the property.")
            return SlopeData(average_slope=0.0, max_slope=0.0)
        
        # Check for elevation column and handle missing case
        if "Elevation" not in intersections.columns:
            print("Elevation column not found in contour data. Using default elevation.")
            elevations = np.array([0.0])  # Default to zero elevation
        else:
            elevations = intersections["Elevation"].values
        print(f"Elevations: {elevations}")
        
        # Calculate slopes between consecutive contour pairs
        slopes = []
        for i in range(len(intersections) - 1):
            elev_diff = abs(elevations[i + 1] - elevations[i])
            geom1 = intersections.iloc[i].geometry
            geom2 = intersections.iloc[i + 1].geometry
            # Calculate distance between centroids of clipped geometries
            centroid1 = geom1.centroid
            centroid2 = geom2.centroid
            dist = centroid1.distance(centroid2)
            bbox1 = geom1.bounds
            bbox2 = geom2.bounds
            geom_wkt = dumps(geom1)
            centroid_coords = (centroid1.y, centroid1.x)  # Latitude, Longitude in projected CRS
            print(f"Pair {i} geometry (WKT): {geom_wkt}")
            print(f"Pair {i} centroid: {centroid_coords}")
            print(f"Pair {i} bounding box: {bbox1}")
            print(f"Pair {i+1} bounding box: {bbox2}")
            print(f"Pair {i}: elev_diff={elev_diff}, dist={dist}")
            # Filter out unrealistic distances (e.g., > 1000 meters or too small)
            if 0.5 < dist <= 1000:
                slope_deg = np.degrees(np.arctan(elev_diff / dist))
                slopes.append(slope_deg)
        
        if not slopes:
            print("No valid slopes calculated after filtering.")
            return SlopeData(average_slope=0.0, max_slope=0.0)
        
        print(f"Slopes: {slopes}")
        return SlopeData(average_slope=np.mean(slopes), max_slope=np.max(slopes))
    except Exception as e:
        print(f"Error calculating slope: {e}")
        return SlopeData(average_slope=0.0, max_slope=0.0)

def check_environmental_hazards(property: Property) -> Optional[EnvironmentalCheck]:
    """
    Check if the property intersects environmental hazard layers.
    
    Args:
        property: Property model instance with geometry.
    
    Returns:
        EnvironmentalCheck model instance indicating hazard presence, or None if check fails.
    """
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
        print(f"Error checking environmental hazards: {e}")
        return None

def create_map(coordinates: Coordinates, property: Property, geojson_files: dict) -> None:
    """
    Create an interactive map with all GeoJSON layers using Folium, each with distinct fill colors.
    
    Args:
        coordinates: Coordinates model instance for map centering.
        property: Property model instance to highlight on the map.
        geojson_files: Dictionary mapping layer names to GeoJSON file paths.
    """
    try:
        # Initialize map centered on the coordinates
        m = folium.Map(location=[coordinates.latitude, coordinates.longitude], zoom_start=15)
        
        # Define color scheme for each layer
        layer_styles = {
            "Property Lines": {"color": "black", "weight": 1, "fill": False},
            "Contours": {"color": "gray", "weight": 1, "fill": False},
            "Erosion Hazard": {"fillColor": "orange", "color": "black", "weight": 1, "fillOpacity": 0.3},
            "Potential Slide Hazard": {"fillColor": "purple", "color": "black", "weight": 1, "fillOpacity": 0.3},
            "Seismic Hazard": {"fillColor": "red", "color": "black", "weight": 1, "fillOpacity": 0.3},
            "Steep Slope Hazard": {"fillColor": "yellow", "color": "black", "weight": 1, "fillOpacity": 0.3},
            "Watercourse Buffer": {"fillColor": "blue", "color": "black", "weight": 1, "fillOpacity": 0.3},
        }

        # Add other GeoJSON layers first (so Property layer renders on top)
        for layer_name, file_path in geojson_files.items():
            gdf = load_geojson(file_path).set_crs("EPSG:4326", allow_override=True)
            style = layer_styles.get(layer_name, {"fillColor": "gray", "color": "black", "weight": 1, "fillOpacity": 0.3})
            folium.GeoJson(
                gdf,
                name=layer_name.capitalize(),
                style_function=lambda x, s=style: s,
                show=True
            ).add_to(m)
        
        # Add Property layer last (to render on top)
        property_geom = shape(property.geometry)
        print(f"Property geometry type: {property_geom.geom_type}")
        # Validate and fix geometry if needed
        if not property_geom.is_valid:
            print("Property geometry is invalid, attempting to fix...")
            property_geom = make_valid(property_geom)
        property_gdf = gpd.GeoDataFrame([property_geom], columns=["geometry"], crs="EPSG:4326")
        folium.GeoJson(
            property_gdf,
            name="Property",
            style_function=lambda x: {
                "fillColor": "green",
                "color": "blue",
                "weight": 2,
                "fillOpacity": 0.5,
                "fill": True  # Explicitly enable fill
            },
            show=True
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display map in Streamlit
        folium_static(m)
    except Exception as e:
        print(f"Error creating map: {e}")