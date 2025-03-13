import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional
from models import LocationAnalysis, SlopeAnalysis, FeasibilityReport
import json

# Ensure the logs directory exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, "geotech_debug.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model with google-generativeai
model = None
try:
    if GOOGLE_API_KEY:
        logging.info(f"GOOGLE_API_KEY found: {GOOGLE_API_KEY[:5]}**** (obfuscated for security)")
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        logging.info("Gemini model initialized successfully with google-generativeai.")
    else:
        logging.warning("GOOGLE_API_KEY not found in environment variables. Skipping Gemini initialization.")
except Exception as e:
    logging.error(f"Failed to initialize Gemini model with google-generativeai: {str(e)}")

# Updated System Prompt
SYSTEM_PROMPT = """
You are an expert geotechnical engineering assistant specializing in property feasibility assessments for Mercer Island, WA. 

CONTEXT:
- Mercer Island has varied topography with steep slopes (>40% in places), seismic concerns (Category D and E), and erosion hazards
- The island is governed by specific municipal codes (MICC 19.07) for environmentally critical areas
- Construction must comply with Washington State landslide hazard regulations (WAC 365-190-120)
- The International Building Code has special requirements for steep slope construction in this region
- The Puget Sound region receives 40+ inches of annual rainfall, creating erosion and drainage concerns

RESPONSE GUIDELINES:
1. Use precise technical terminology from geotechnical engineering
2. Prioritize risks by severity: critical (safety/structural concerns), major (significant cost/design impacts), minor (manageable with standard techniques)
3. Include feasibility assessment with specific regulatory references
4. Ensure all recommendations mention both priority level AND cost implication within each string
5. When estimating costs, use relative terms (low, moderate, high) rather than dollar figures

OUTPUT FORMAT:
- Return all responses as properly structured JSON objects 
- Enclose JSON in triple backticks (```json ... ```)
- If data is missing, clearly state assumptions based on Mercer Island averages
- Format recommendation strings as "[Priority Level]: [Recommendation text] ([cost level] cost)"
"""

def parse_gemini_json_response(response_text: str) -> dict:
    """Helper function to parse Gemini response wrapped in triple backticks."""
    response_text = response_text.strip()
    if response_text.startswith("```json") and response_text.endswith("```"):
        json_str = response_text[len("```json"):].rsplit("```", 1)[0].strip()
        return json.loads(json_str)
    else:
        raise ValueError(f"Response is not in expected JSON format: {response_text}")

def analyze_location(latitude: float, longitude: float, address: str) -> Optional[LocationAnalysis]:
    """Analyze the location using Gemini with updated prompt."""
    if model is None:
        logging.warning("Skipping location analysis due to missing or failed Gemini initialization.")
        return None

    prompt = f"""
{SYSTEM_PROMPT}

TASK: Analyze property location at latitude {latitude}, longitude {longitude}, address '{address}'

LOCATION ANALYSIS CONSIDERATIONS:
- Geographic position relative to Mercer Island's known geological features
- Proximity to water bodies, slopes, and potential landslide areas
- Construction access challenges (steep driveways, tight corners, etc.)
- Neighborhood context and property value implications
- Historical issues at similar locations on Mercer Island

REGIONAL CONTEXT:
- Eastern shore: Generally steeper slopes and erosion concerns
- Central plateau: More stable but may have isolated drainage issues
- Western shores: May have issues with wave action and shoreline stability
- Northern areas: May experience higher vibration from I-90 traffic
- Southern areas: Often have more stringent tree preservation requirements

EXPECTED OUTPUT STRUCTURE:
{{
  "summary": "A comprehensive 3-5 sentence technical assessment of the location",
  "recommendations": [
    "Critical: [First recommendation focused on highest priority issue] (high cost)",
    "Major: [Second recommendation focused on significant issue] (moderate cost)",
    "Minor: [Additional recommendation] (low cost)",
    ...
  ]
}}
"""
    try:
        response = model.generate_content(prompt)
        logging.info(f"Raw location analysis response: {response.text[:100]}...")
        response_data = parse_gemini_json_response(response.text)
        return LocationAnalysis(**response_data)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse location analysis response as JSON: {str(e)}. Raw response: {response.text}")
        return None
    except ValueError as e:
        logging.error(f"Invalid response format: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error during location analysis: {str(e)}")
        return None

def analyze_slope(slope: float, elevation_diff: float, distance: float) -> Optional[SlopeAnalysis]:
    """Analyze the slope using Gemini with updated prompt."""
    if model is None:
        logging.warning("Skipping slope analysis due to missing or failed Gemini initialization.")
        return None

    prompt = f"""
{SYSTEM_PROMPT}

TASK: Analyze slope profile with {slope:.2f}% slope, {elevation_diff:.2f} ft elevation difference, {distance:.2f} m distance

SLOPE CLASSIFICATION REFERENCE:
- Mild: <15% (Minimal restrictions)
- Moderate: 15-25% (Standard engineering solutions required)
- Steep: 25-40% ("Steep Slope Hazard Area" under MICC 19.07.100, requiring geotechnical report)
- Very steep: >40% ("Very Steep Slope Hazard Area" with substantial restrictions)

MERCER ISLAND SOIL TYPES AND STABILITY:
- Glacial till: Relatively stable, dense, low permeability
- Vashon advance outwash: Moderately stable, variable density
- Recessional outwash: Less stable, more permeable
- Fill: Potentially unstable, requiring thorough assessment

FOUNDATION CONSIDERATIONS FOR THIS SLOPE:
- Conventional spread footings: Suitable for slopes <15%
- Stepped foundations: Often used for 15-25% slopes
- Pin piles: Commonly needed for 25-40% slopes
- Caissons: May be required for slopes >40%
- Retaining structures: Engineering requirements scale with slope percentage

EXPECTED OUTPUT STRUCTURE:
{{
  "summary": "A technically precise assessment of engineering challenges based on slope characteristics",
  "recommendations": [
    "Major: [Foundation recommendation] (moderate cost)",
    "Critical: [Drainage recommendation] (high cost)",
    "Minor: [Erosion control recommendation] (low cost)",
    ...
  ]
}}
"""
    try:
        response = model.generate_content(prompt)
        logging.info(f"Raw slope analysis response: {response.text[:100]}...")
        response_data = parse_gemini_json_response(response.text)
        return SlopeAnalysis(**response_data)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse slope analysis response as JSON: {str(e)}. Raw response: {response.text}")
        return None
    except ValueError as e:
        logging.error(f"Invalid response format: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error during slope analysis: {str(e)}")
        return None

def generate_feasibility_report(
    address: str,
    slope_analysis: Optional[SlopeAnalysis],
    location_analysis: Optional[LocationAnalysis],
    environmental_hazards: dict,
) -> Optional[FeasibilityReport]:
    """Generate a feasibility report using Gemini with updated prompt."""
    if model is None:
        logging.warning("Skipping feasibility report generation due to missing or failed Gemini initialization.")
        return None

    # Format hazard layer information
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

TASK: Generate a comprehensive feasibility report for property at '{address}'

INPUT DATA:
1. Location Analysis: {location_analysis.dict() if location_analysis else "No location analysis available."}
2. Slope Analysis: {slope_analysis.dict() if slope_analysis else "No slope analysis available."}
3. Environmental Hazards: {json.dumps(environmental_hazards, indent=2)}
4. Hazard Layer Information: {json.dumps(hazard_layer_list, indent=2)}

FEASIBILITY ASSESSMENT FRAMEWORK:
- TECHNICAL FACTORS: Slope stability, soil conditions, drainage requirements
- REGULATORY FACTORS: Compliance with MICC 19.07, setback requirements, environmental mitigation
- ECONOMIC FACTORS: Relative costs compared to typical Mercer Island development
- TIMELINE FACTORS: Permit process, seasonal construction limitations, specialist availability

FEASIBILITY CLASSIFICATIONS:
1. "Not Feasible": Critical barriers that likely prevent development (technical/regulatory showstoppers)
2. "Marginally Feasible": Significant challenges requiring specialized engineering solutions (>50% cost premium)
3. "Moderately Feasible": Notable challenges with established mitigation approaches (20-50% cost premium)
4. "Highly Feasible": Minimal challenges compared to typical Mercer Island properties (<20% cost premium)

EXPECTED OUTPUT STRUCTURE:
{{
  "location_analysis": {{ 
    "summary": "...", 
    "recommendations": ["...", "..."] 
  }},
  "slope_analysis": {{ 
    "summary": "...", 
    "recommendations": ["...", "..."] 
  }},
  "overall_feasibility": "One of the four classifications defined above",
  "detailed_recommendations": [
    "Critical: [First implementation step] (high cost)",
    "Major: [Second implementation step] (moderate cost)",
    "Minor: [Additional consideration] (low cost)",
    ...
  ],
  "hazard_layers": [Use verbatim from provided hazard_layer_list]
}}
"""
    try:
        response = model.generate_content(prompt)
        logging.info(f"Raw feasibility report response: {response.text[:100]}...")
        response_data = parse_gemini_json_response(response.text)
        return FeasibilityReport(**response_data)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse feasibility report response as JSON: {str(e)}. Raw response: {response.text}")
        return None
    except ValueError as e:
        logging.error(f"Invalid response format: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error during feasibility report generation: {str(e)}")
        return None

def chat_with_report(report: FeasibilityReport, user_query: str) -> Optional[str]:
    """Chat with the user about the feasibility report using Gemini with updated prompt."""
    if model is None:
        logging.warning("Skipping chat due to missing or failed Gemini initialization.")
        return None

    prompt = f"""
{SYSTEM_PROMPT}

TASK: Respond to a property owner's question about their feasibility report

FEASIBILITY REPORT:
{json.dumps(report.dict(), indent=2)}

USER QUESTION:
"{user_query}"

RESPONSE GUIDELINES:
1. Prioritize clarity: Use plain language first, technical terms only when necessary
2. When using technical terms, briefly define them in parentheses
3. Reference specific data from the report to support your answer
4. If the question exceeds the report's scope, acknowledge the limitation and suggest what additional information might help
5. For regulatory questions, cite specific codes (e.g., "According to MICC 19.07.120...")
6. For cost questions, provide ranges rather than specific figures, noting that exact costs require contractor bids
7. Maintain objectivity without being overly optimistic or pessimistic about development prospects

Your response should be helpful, informative, and directly address the user's specific question using a professional but conversational tone appropriate for a property owner who may not have technical engineering background.
"""
    try:
        response = model.generate_content(prompt)
        logging.info(f"Raw chat response: {response.text[:100]}...")
        return response.text
    except Exception as e:
        logging.error(f"Error during chat with report: {str(e)}")
        return None