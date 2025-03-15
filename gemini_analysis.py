import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, List
from models import LocationAnalysis, SlopeAnalysis, FeasibilityReport, SlopeData
import json
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
else:
    logging.warning("GOOGLE_API_KEY not found. Gemini model not initialized.")
    model = None

# Define the enhanced system prompt
SYSTEM_PROMPT = """
You are GeotechExpert, a geotechnical engineering AI assistant with deep expertise in Mercer Island, WA, properties, designed to deliver precise, data-driven feasibility assessments comparable to a seasoned professional with 25 years of regional experience.

1. ROLE:
- Analyze geotechnical factors (e.g., slope stability, soil conditions, liquefaction potential) using provided data, enriched with Mercer Island-specific knowledge (e.g., glacial till over lacustrine deposits, lakefront erosion dynamics).
- Use precise geotechnical terminology (e.g., 'factor of safety', 'angle of repose') and reference regional standards (e.g., MICC 19.07, USGS seismic data).

2. BOUNDARIES:
- DO NOT provide legal advice, specific cost estimates (e.g., dollar amounts), or non-geotechnical information unless explicitly tied to feasibility (e.g., permitting implications).
- DO NOT speculate beyond data; if data is missing, use Mercer Island norms (e.g., 'Assuming glacial till with 30° angle of repose unless verified') and flag verification needs.
- For out-of-scope queries, respond: 'I'm limited to geotechnical analysis and cannot assist with [legal/cost] matters without additional context.'

3. ANALYSIS GUIDELINES:
- Contextualize findings with Mercer Island's geology (e.g., steep lakefront slopes, saturated soils).
- Flag anomalies (e.g., slopes >40% with no slide history) with 'Potential inconsistency—verify data.'
- Assign confidence levels based on data quality: High (complete data), Medium (partial data with norms), Low (assumptions only).
- Integrate hazards into a cohesive assessment (e.g., erosion + steep slope = compounded slide risk).

4. RESPONSE GUIDELINES:
- Return structured JSON in triple backticks (```json ... ```).
- Format recommendations as '[Priority]: [Text] ([cost level] cost) - Confidence: [Level]' using [Critical], [Major], or [Minor], with relative costs (low, moderate, high).
- Provide specific, actionable steps (e.g., 'Conduct 3-4 borings to 20 ft depth') over generic advice.
- Include a 'verification_needed' field if assumptions are made (e.g., soil type).

5. USER ADAPTATION:
- For chat responses, detect user expertise (technical vs. layperson) from query phrasing and adjust tone: technical (e.g., 'Liquefaction potential requires SPT data') or plain (e.g., 'The ground might shift in an earthquake—more tests needed').
"""

def parse_gemini_json_response(response_text: str) -> dict:
    """Parse the JSON response from the Gemini model, normalizing priority levels."""
    response_text = response_text.strip()
    if response_text.startswith("```json") and response_text.endswith("```"):
        json_str = response_text[len("```json"):].rsplit("```", 1)[0].strip()
        response_data = json.loads(json_str)
        # Normalize recommendation priorities
        for section in ["recommendations", "detailed_recommendations"]:
            if section in response_data:
                for i, rec in enumerate(response_data[section]):
                    if rec.startswith("[High]"):
                        response_data[section][i] = rec.replace("[High]", "[Critical]")
                    elif rec.startswith("[Medium]"):
                        response_data[section][i] = rec.replace("[Medium]", "[Major]")
        if "location_analysis" in response_data and "recommendations" in response_data["location_analysis"]:
            for i, rec in enumerate(response_data["location_analysis"]["recommendations"]):
                if rec.startswith("[High]"):
                    response_data["location_analysis"]["recommendations"][i] = rec.replace("[High]", "[Critical]")
                elif rec.startswith("[Medium]"):
                    response_data["location_analysis"]["recommendations"][i] = rec.replace("[Medium]", "[Major]")
        if "slope_analysis" in response_data and "recommendations" in response_data["slope_analysis"]:
            for i, rec in enumerate(response_data["slope_analysis"]["recommendations"]):
                if rec.startswith("[High]"):
                    response_data["slope_analysis"]["recommendations"][i] = rec.replace("[High]", "[Critical]")
                elif rec.startswith("[Medium]"):
                    response_data["slope_analysis"]["recommendations"][i] = rec.replace("[Medium]", "[Major]")
        return response_data
    else:
        raise ValueError(f"Response is not in expected JSON format: {response_text}")

def analyze_location(latitude: float, longitude: float, address: str, hazards: dict, lake_proximity: bool) -> Optional[LocationAnalysis]:
    """Analyze the geotechnical aspects of a property location with Mercer Island context."""
    if model is None:
        logging.warning("Gemini model not initialized. Skipping location analysis.")
        return None

    prompt = f"""
{SYSTEM_PROMPT}

TASK: Analyze property location at latitude {latitude}, longitude {longitude}, address '{address}'

KEY DATA SUMMARY:
- Coordinates: {latitude}, {longitude}
- Address: {address}
- Environmental Hazards: {json.dumps(hazards)}
- Lake Proximity (within 100m of erosion hazard): {lake_proximity}

REASONING PROCESS:
1. Identify topographic position (e.g., lakefront, plateau) using coordinates and hazard data.
2. Assess proximity to geological features (e.g., Lake Washington shoreline <100m = wave erosion risk) if lake_proximity or erosion hazard is true.
3. Evaluate access constraints only if hazards suggest steep terrain or watercourse issues.
4. If no hazards and not near lakefront, prioritize minimal investigation.

INSTRUCTIONS:
- Use hazard data to tailor recommendations (e.g., erosion present = silt fences).
- Recommend specific tests only if risks are indicated (e.g., 'Borings to 20 ft' for seismic or lakefront).
- Output:
```json
{{
  "summary": "Property at {address} shows flat topography and no immediate hazard risks based on data.",
  "recommendations": [
    "[Minor]: Conduct shallow borings to confirm soil stability (low cost) - Confidence: High"
  ],
  "verification_needed": ["Soil type"]
}}
"""
    try:
        logging.debug(f"Location Analysis Prompt: {prompt}")
        response = model.generate_content(prompt)
        logging.debug(f"Location Analysis Response: {response.text}")
        response_data = parse_gemini_json_response(response.text)
        return LocationAnalysis(**response_data)
    except Exception as e:
        logging.error(f"Error in analyze_location: {e}")
        return None

def analyze_slope(slope: float, elevation_diff: float, distance: float, lake_proximity: bool) -> Optional[SlopeAnalysis]:
    """Analyze the slope profile of a property with detailed stability assessment."""
    if model is None:
        logging.warning("Gemini model not initialized. Skipping slope analysis.")
        return None
    
    prompt = f"""
{SYSTEM_PROMPT}

TASK: Analyze slope profile with {slope:.2f}° slope, {elevation_diff:.2f} m elevation difference, {distance:.2f} m distance

KEY DATA SUMMARY:
- Slope: {slope:.2f}°
- Elevation Difference: {elevation_diff:.2f} m
- Distance: {distance:.2f} m
- Lake Proximity (within 100m of erosion hazard): {lake_proximity}

REASONING PROCESS:
1. Classify slope per Mercer Island standards: <15° mild, 15-25° moderate, 25-40° steep, >40° very steep (exceeds glacial till repose).
2. Assess stability using typical soil types (glacial till = 30° repose, lacustrine = 20° repose) unless specified.
3. If slope <5° but lake_proximity is true, flag 'Potential inconsistency—verify topographic survey near lakefront.'
4. Recommend solutions based on slope and soil (e.g., >40° = soldier piles, <25° = drainage).

INSTRUCTIONS:
- Assume glacial till unless contradicted; flag: 'Verification Needed: Soil layering.'
- Specify test locations (e.g., '3-4 borings across slope') and methods (e.g., 'Direct shear testing').
- Output:
```json
{{
  "summary": "Slope is flat (<5°), but lakefront proximity suggests potential instability—verify data.",
  "recommendations": [
    "[Minor]: Verify soil with shallow borings (low cost) - Confidence: High"
  ],
  "verification_needed": ["Topographic survey near lakefront", "Soil type"]
}}
"""
    try:
        logging.debug(f"Slope Analysis Prompt: {prompt}")
        response = model.generate_content(prompt)
        logging.debug(f"Slope Analysis Response: {response.text}")
        response_data = parse_gemini_json_response(response.text)
        return SlopeAnalysis(**response_data)
    except Exception as e:
        logging.error(f"Error in analyze_slope: {e}")
        return None

def generate_feasibility_report(
    address: str,
    slope_analysis: Optional[SlopeAnalysis],
    location_analysis: Optional[LocationAnalysis],
    environmental_hazards: dict,
    slope_data: SlopeData,
    lake_proximity: bool
) -> Optional[FeasibilityReport]:
    """Generate a comprehensive feasibility report with integrated hazard analysis."""
    if model is None:
        logging.warning("Gemini model not initialized. Skipping feasibility report generation.")
        return None
    
    # Define hazard descriptions and create hazard layer list
    hazard_descriptions = {
        "erosion": "Erosion Hazard",
        "potential_slide": "Potential Slide Hazard",
        "seismic": "Seismic Hazard",
        "steep_slope": "Steep Slope Hazard",
        "watercourse": "Watercourse Buffer"
    }
    
    hazard_layer_list = [
        f"{hazard_descriptions[key]}: {'Present' if value else 'Not Present'} - Property {'falls within' if value else 'does not fall within'} a {hazard_descriptions[key]}"
        for key, value in environmental_hazards.items()
    ]
    
    prompt = f"""
{SYSTEM_PROMPT}

TASK: Generate a comprehensive feasibility report for '{address}', integrating hazards and practical refinements.

KEY DATA SUMMARY:
- Address: {address}
- Location Analysis: {location_analysis.dict() if location_analysis else 'No data'}
- Slope Analysis: {slope_analysis.dict() if slope_analysis else 'No data'}
- Slope Data: Avg {slope_data.average_slope}°, Max {slope_data.max_slope}°, Avg Distance {slope_data.average_distance}m
- Environmental Hazards: {json.dumps(environmental_hazards)}
- Lake Proximity (within 100m of erosion hazard): {lake_proximity}

FRAMEWORK:
- Technical: Assess slope stability, soil, drainage only if data indicates risks (e.g., slope >15° or hazards present).
- Regulatory: Reference MICC 19.07 only for relevant mitigations.
- Practical: Sequence construction conservatively only if erosion or slope risks are present.
- Hazards: Link recommendations to data; if no hazards and slope <5°, classify as Highly Feasible unless contradicted by soil or access issues.

INSTRUCTIONS:
- Classify feasibility: Not Feasible (<30% success), Marginally Feasible (30-50%), Moderately Feasible (50-75%), Highly Feasible (>75%). Use Highly Feasible if slope <5° and no hazards unless contradicted by soil or access issues.
- Suggest minimal foundations (e.g., shallow spread footings) for flat, hazard-free sites.
- Output:
```json
{{
  "location_analysis": {location_analysis.dict() if location_analysis else '{"summary": "Pending", "recommendations": [], "verification_needed": []}'},
  "slope_analysis": {slope_analysis.dict() if slope_analysis else '{"summary": "Pending", "recommendations": [], "verification_needed": []}'},
  "overall_feasibility": "Highly Feasible (>75% success)",
  "detailed_recommendations": [
    "[Minor]: Use shallow spread footings for foundation (low cost) - Confidence: High",
    "[Minor]: Verify soil bearing capacity with shallow borings (low cost) - Confidence: High"
  ],
  "hazard_layers": {json.dumps(hazard_layer_list)},
  "verification_needed": ["Soil bearing capacity"]
}}
"""
    try:
        logging.debug(f"Feasibility Report Prompt: {prompt}")
        response = model.generate_content(prompt)
        logging.debug(f"Feasibility Report Response: {response.text}")
        response_data = parse_gemini_json_response(response.text)
        return FeasibilityReport(**response_data)
    except Exception as e:
        logging.error(f"Error in generate_feasibility_report: {e}")
        return None

def chat_with_report(report: FeasibilityReport, user_query: str, chat_history: List[tuple]) -> Optional[str]:
    """Respond to user queries about the feasibility report with adaptive tone."""
    if model is None:
        logging.warning("Gemini model not initialized. Skipping chat response.")
        return None
    
    history_str = json.dumps([q for q, _ in chat_history], indent=2) if chat_history else 'No prior questions.'
    
    prompt = f"""
{SYSTEM_PROMPT}

TASK: Respond to '{user_query}' about the feasibility report.

FEASIBILITY REPORT:
{json.dumps(report.dict())}

HISTORY:
{history_str}

INSTRUCTIONS:
- Detect expertise: technical queries (e.g., 'What's FOS?') get detailed terms; layperson (e.g., 'Is it safe?') get plain language + terms explained.
- Reference report specifics (e.g., 'Your 40° slope exceeds till repose').
- Suggest practical next steps (e.g., 'Engage a geotech for borings').
- Output plain text unless JSON requested: "Technical: Your slope's factor of safety (FOS, stability measure) may be <1.5 without piles—recommend 3 borings. Layperson: The steep hill might slide; get a soil test soon."
- Add: 'If this response seems inaccurate, please flag it for review.'
"""
    try:
        logging.debug(f"Chat Prompt: {prompt}")
        response = model.generate_content(prompt)
        logging.debug(f"Chat Response: {response.text}")
        return response.text
    except Exception as e:
        logging.error(f"Error in chat_with_report: {e}")
        return None