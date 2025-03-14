import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, List
from models import LocationAnalysis, SlopeAnalysis, FeasibilityReport
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
You are GeotechExpert, a geotechnical engineering AI assistant specialized in Mercer Island properties, designed to deliver accurate, data-driven feasibility assessments.

1. ROLE:
- Analyze geotechnical factors (e.g., slope stability, soil conditions) based solely on provided input data.
- Use precise geotechnical terms (e.g., 'factor of safety', 'liquefaction potential').

2. BOUNDARIES:
- DO NOT provide legal advice, specific cost estimates (e.g., dollar amounts), or non-geotechnical information.
- DO NOT speculate beyond the provided data; if data is missing, use Mercer Island averages and state assumptions explicitly (e.g., 'Assuming glacial till soil due to regional norms').
- For out-of-scope queries, respond: 'I'm limited to geotechnical analysis and cannot assist with [legal/cost] matters.'

3. RESPONSE GUIDELINES:
- Return structured JSON in triple backticks (```json ... ```).
- Format recommendations as '[Priority]: [Text] ([cost level] cost)' using only [Critical], [Major], or [Minor] as priorities, with relative costs (low, moderate, high).
- Include a confidence level (High, Medium, Low) for each recommendation, based on data sufficiency.
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

def analyze_location(latitude: float, longitude: float, address: str) -> Optional[LocationAnalysis]:
    """Analyze the geotechnical aspects of a property location."""
    if model is None:
        logging.warning("Gemini model not initialized. Skipping location analysis.")
        return None

    prompt = f"""
{SYSTEM_PROMPT}

TASK: Analyze property location at latitude {latitude}, longitude {longitude}, address '{address}'

KEY DATA SUMMARY:
- Coordinates: {latitude}, {longitude}
- Address: {address}

REASONING PROCESS:
1. Determine the property's position on Mercer Island (e.g., eastern shore, central plateau).
2. Assess proximity to known geological features (e.g., water bodies, steep slopes).
3. Evaluate construction access based on location.
4. Consider neighborhood context and historical issues.

INSTRUCTIONS:
- Base your analysis strictly on the provided coordinates and address.
- If specific data is missing (e.g., exact distance to water), use regional norms and state assumptions (e.g., 'Assuming typical distance for eastern shore properties').
- Use only [Critical], [Major], or [Minor] as recommendation priorities.
- Include confidence levels for each recommendation (High, Medium, Low).
- If your assessment seems inconsistent (e.g., a central plateau property with steep slope concerns), flag it as 'Potential inconsistency—please verify data.'

EXPECTED OUTPUT STRUCTURE:
```json
{{
  "summary": "A 3-5 sentence technical assessment",
  "recommendations": [
    "[Critical]: Assess erosion risk near shore (high cost) - Confidence: High",
    "[Minor]: Ensure driveway access (low cost) - Confidence: Medium"
  ]
}}
"""
    try:
        response = model.generate_content(prompt)
        response_data = parse_gemini_json_response(response.text)
        return LocationAnalysis(**response_data)
    except Exception as e:
        logging.error(f"Error in analyze_location: {e}")
        return None

def analyze_slope(slope: float, elevation_diff: float, distance: float) -> Optional[SlopeAnalysis]:
    """Analyze the slope profile of a property."""
    if model is None:
        logging.warning("Gemini model not initialized. Skipping slope analysis.")
        return None
    
    prompt = f"""
{SYSTEM_PROMPT}

TASK: Analyze slope profile with {slope:.2f}% slope, {elevation_diff:.2f} ft elevation difference, {distance:.2f} m distance

KEY DATA SUMMARY:
- Slope: {slope:.2f}% (equivalent to {np.degrees(np.arctan(slope / 100)):.2f} degrees)
- Elevation Difference: {elevation_diff:.2f} ft
- Distance: {distance:.2f} m

REASONING PROCESS:
1. Classify the slope using Mercer Island standards (<15% mild, 15-25% moderate, 25-40% steep, >40% very steep).
2. Evaluate stability based on typical Mercer Island soil types (e.g., glacial till, outwash).
3. Recommend foundation or drainage solutions based on slope severity.

INSTRUCTIONS:
- Base each step explicitly on the provided slope, elevation difference, and distance data.
- If soil type is unknown, assume glacial till and state: 'Assuming glacial till soil due to regional norms.'
- Provide interim conclusions after each step.
- Use only [Critical], [Major], or [Minor] as recommendation priorities.
- Include confidence levels for recommendations (e.g., 'High confidence due to clear slope data').

EXPECTED OUTPUT STRUCTURE:
```json
{{
  "summary": "A 3-5 sentence assessment with reasoning steps",
  "recommendations": [
    "[Major]: Install retaining wall (moderate cost) - Confidence: High",
    "[Minor]: Monitor drainage (low cost) - Confidence: Medium"
  ]
}}
"""
    try:
        response = model.generate_content(prompt)
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
) -> Optional[FeasibilityReport]:
    """Generate a comprehensive feasibility report for a property."""
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

TASK: Generate a comprehensive feasibility report for property at '{address}', explicitly integrating the provided environmental hazard layer information into the analysis and recommendations.

KEY DATA SUMMARY:
- Address: {address}
- Location Analysis: {'Completed' if location_analysis else 'Pending'}
- Slope Analysis: {'Completed' if slope_analysis else 'Pending'}
- Environmental Hazards: {json.dumps(environmental_hazards, indent=2)}

INPUT DATA:
- Location Analysis: {location_analysis.dict() if location_analysis else 'No location analysis available.'}
- Slope Analysis: {slope_analysis.dict() if slope_analysis else 'No slope analysis available.'}
- Hazard Layer Information: {json.dumps(hazard_layer_list, indent=2)}

FEASIBILITY ASSESSMENT FRAMEWORK:
- TECHNICAL FACTORS: Slope stability, soil conditions, drainage requirements
- REGULATORY FACTORS: Compliance with MICC 19.07, setback requirements, environmental mitigation
- ECONOMIC FACTORS: Relative costs compared to typical Mercer Island development
- TIMELINE FACTORS: Permit process, seasonal construction limitations, specialist availability
- ENVIRONMENTAL HAZARD FACTORS: Explicitly assess the impact of each hazard layer (erosion, potential slide, seismic, steep slope, watercourse) on feasibility, referencing their presence or absence in the summary and recommendations

VISUAL INTEGRATION:
- Reference map visuals in recommendations (e.g., 'Given the red seismic hazard zone on the map, consider...', 'Due to the orange steep slope area, implement...').

INSTRUCTIONS:
- Base your assessment strictly on the provided analyses and hazard data.
- If data is missing, use Mercer Island averages and state assumptions (e.g., 'Assuming typical soil conditions for the area').
- Use only [Critical], [Major], or [Minor] as recommendation priorities.
- Include confidence levels for each recommendation.
- Provide an executive summary for non-technical stakeholders, followed by detailed technical recommendations.

EXPECTED OUTPUT STRUCTURE:
```json
{{
  "location_analysis": {{ 
    "summary": "...", 
    "recommendations": ["...", "..."] 
  }},
  "slope_analysis": {{ 
    "summary": "...", 
    "recommendations": ["...", "..."] 
  }},
  "overall_feasibility": "One of the four classifications: Not Feasible, Marginally Feasible, Moderately Feasible, Highly Feasible",
  "detailed_recommendations": [
    "[Critical]: [First implementation step] (high cost) - Confidence: High",
    "[Major]: [Second implementation step] (moderate cost) - Confidence: Medium"
  ],
  "hazard_layers": {json.dumps(hazard_layer_list, indent=2)}
}}
"""
    try:
        response = model.generate_content(prompt)
        response_data = parse_gemini_json_response(response.text)
        return FeasibilityReport(**response_data)
    except Exception as e:
        logging.error(f"Error in generate_feasibility_report: {e}")
        return None

def chat_with_report(report: FeasibilityReport, user_query: str, chat_history: List[tuple]) -> Optional[str]:
    """Respond to user queries about the feasibility report."""
    if model is None:
        logging.warning("Gemini model not initialized. Skipping chat response.")
        return None
    
    history_str = json.dumps([q for q, _ in chat_history], indent=2) if chat_history else 'No prior questions.'
    
    prompt = f"""
{SYSTEM_PROMPT}

TASK: Respond to the user's question about their feasibility report.

FEASIBILITY REPORT:
{json.dumps(report.dict(), indent=2)}

USER QUESTION:
"{user_query}"

CONVERSATION HISTORY:
{history_str}

INSTRUCTIONS:
- Reference the feasibility report and previous questions to maintain consistency.
- Use plain language first, then add technical terms with brief explanations (e.g., 'factor of safety' = stability measure).
- If the question exceeds the report's scope, acknowledge the limitation and suggest additional information needed.
- For regulatory questions, cite specific codes (e.g., "According to MICC 19.07.120...").
- For cost questions, provide relative terms (low, moderate, high) without specific figures.
- Include confidence levels where applicable (e.g., 'High confidence based on provided data').
- If data is insufficient, state: 'Insufficient data—recommend further investigation.'
- Encourage feedback by adding: 'If this response seems inaccurate, please flag it for review.'
- Your response should be helpful, informative, and directly address the user's question in a professional but conversational tone.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error in chat_with_report: {e}")
        return None