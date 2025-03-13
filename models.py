from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class Address(BaseModel):
    """Model for user-input address."""
    street: str = Field(..., description="Street address")
    city: str = Field(default="Mercer Island", description="City (fixed to Mercer Island)")
    state: str = Field(default="WA", description="State (fixed to WA)")
    zip_code: Optional[str] = Field(None, description="ZIP code (optional)")

    def full_address(self) -> str:
        """Return the full address string for geocoding."""
        parts = [self.street, self.city, self.state]
        if self.zip_code:
            parts.append(self.zip_code)
        return ", ".join(parts)

class Coordinates(BaseModel):
    """Model for geocoded coordinates."""
    latitude: float = Field(..., description="Latitude in degrees")
    longitude: float = Field(..., description="Longitude in degrees")

class Property(BaseModel):
    """Model for property data extracted from GeoJSON."""
    parcel_id: str = Field(..., description="Unique parcel identifier")
    geometry: dict = Field(..., description="GeoJSON geometry of the property")

class SlopeData(BaseModel):
    """Model for slope data extracted from contours."""
    average_slope: float = Field(..., description="Average slope in degrees")
    max_slope: float = Field(..., description="Maximum slope in degrees")

class EnvironmentalCheck(BaseModel):
    """Model for environmental hazard checks."""
    erosion: bool = Field(..., description="Property intersects erosion hazard")
    potential_slide: bool = Field(..., description="Property intersects potential slide area")
    seismic: bool = Field(..., description="Property intersects seismic hazard")
    steep_slope: bool = Field(..., description="Property intersects steep slope hazard")
    watercourse: bool = Field(..., description="Property intersects watercourse buffer")

class LocationAnalysis(BaseModel):
    """Model for Gemini location analysis output."""
    summary: str = Field(..., description="Summary of environmental risks")
    recommendations: List[str] = Field(..., description="List of mitigation recommendations")

class SlopeAnalysis(BaseModel):
    """Model for Gemini slope analysis output."""
    summary: str = Field(..., description="Summary of slope stability")
    recommendations: List[str] = Field(..., description="List of slope-related recommendations")

class FeasibilityReport(BaseModel):
    """Model for the final feasibility report."""
    location_analysis: LocationAnalysis = Field(..., description="Location analysis results")
    slope_analysis: SlopeAnalysis = Field(..., description="Slope analysis results")
    overall_feasibility: str = Field(..., description="Overall feasibility assessment")
    detailed_recommendations: List[str] = Field(..., description="Detailed recommendations for development")
    hazard_layers: List[str] = Field(..., description="List of hazard layer statuses (e.g., 'Erosion Hazard: Present - Property falls within an Erosion Hazard')")