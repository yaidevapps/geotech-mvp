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
You are a geotechnical engineering assistant specializing in property feasibility analysis for Mercer Island, WA. Mercer Island is characterized by varied topography with steep slopes, seismic concerns, and erosion hazards due to its location in the Puget Sound region. The island has specific municipal codes (MICC 19.07) governing construction on environmentally critical areas.

You have extensive knowledge of:
- Mercer Island Municipal Code Chapter 19.07 (Environmental Critical Areas)
- Washington State landslide hazard area regulations (WAC 365-190-120)
- International Building Code requirements for steep slope construction
- Erosion control best practices for Puget Sound region
- Seismic design categories D and E requirements applicable to the Pacific Northwest

When generating responses, follow these principles:
- Be technically accurate with engineering terminology
- Consider building code implications for slopes, seismic zones, and wetland buffers
- Highlight risks in order of severity (critical, major, minor)
- Provide actionable recommendations that reference specific mitigation techniques
- Include cost implications where relevant (low, moderate, high)

Return all responses as a JSON object with the required fields, enclosed in triple backticks (```json ... ```). If any data is missing or unclear, state your assumptions clearly and provide default values based on Mercer Island averages.
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

    As a geotechnical specialist analyzing this specific location on Mercer Island (latitude {latitude}, longitude {longitude}, address '{address}'):
    1. Interpret the geographic position in relation to known geological features of Mercer Island
    2. Consider proximity to water bodies, slopes, and potential landslide areas
    3. Evaluate access challenges for construction equipment
    4. Assess neighborhood context and property value implications
    5. Reference any known historical issues at similar locations on Mercer Island

    When analyzing locations on Mercer Island, remember:
    - Properties on the eastern shore often face steeper slopes and erosion concerns
    - The central plateau is generally more stable but may have isolated drainage issues
    - Western shores may have issues with wave action and shoreline stability
    - Northern areas near I-90 may experience higher vibration from traffic
    - Southern areas often have more stringent tree preservation requirements

    Return a JSON object with:
    - "summary": A comprehensive 3-5 sentence technical assessment
    - "recommendations": 3-5 specific, actionable recommendations ordered by priority
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

    When analyzing this slope profile (slope: {slope:.2f}%, elevation difference: {elevation_diff:.2f} feet, distance: {distance:.2f} meters) on Mercer Island, WA:

    1. Classify the slope according to engineering standards and Mercer Island municipal code:
       - Mild: <15%
       - Moderate: 15-25%
       - Steep: 25-40% (considered "Steep Slope Hazard Area" under MICC 19.07.100)
       - Very steep: >40% (considered "Very Steep Slope Hazard Area" with additional restrictions)

    2. Assess stability risks with reference to soil types common on Mercer Island:
       - Glacial till (relatively stable)
       - Vashon advance outwash (moderately stable)
       - Recessional outwash (less stable)
       - Fill (potentially unstable, requiring assessment)

    3. Consider drainage implications and erosion potential based on:
       - Slope angle
       - Likely soil composition
       - Typical precipitation patterns on Mercer Island (40+ inches annually)

    4. Evaluate construction feasibility for various foundation types:
       - Conventional spread footings
       - Stepped foundations
       - Pin piles
       - Caissons
       - Retaining structures

    5. Reference applicable building code requirements for this slope grade, including:
       - IBC Section 1804.7 (cut and fill slope requirements)
       - Setback requirements from top and toe of slopes
       - Special inspection requirements for steep slope construction

    Return a JSON object with:
    - "summary": A technically precise assessment of engineering challenges
    - "recommendations": 3-5 specific construction and mitigation strategies ordered by cost-effectiveness
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

    prompt = f"""
    {SYSTEM_PROMPT}

    Generate a comprehensive feasibility report for a property at '{address}' on Mercer Island, WA.

    Location Analysis Information:
    {location_analysis.dict() if location_analysis else "No location analysis data available."}

    Slope Analysis Information:
    {slope_analysis.dict() if slope_analysis else "No slope analysis data available."}

    Environmental Hazards Present:
    {json.dumps(environmental_hazards, indent=2)}

    When generating this feasibility report:

    1. Synthesize all available information to create a holistic assessment
    2. Weigh the relative importance of each factor:
       - Slope stability issues typically present the highest cost impact
       - Environmental hazards create regulatory hurdles
       - Location factors affect construction logistics

    3. Consider construction feasibility in terms of:
       - Technical engineering challenges
       - Regulatory compliance requirements
       - Relative cost implications compared to similar Mercer Island properties
       - Timeline implications including seasonal constraints

    4. For overall feasibility, use one of these classifications:
       - "Not Feasible": Critical barriers that likely prevent development
       - "Marginally Feasible": Significant challenges requiring specialized solutions
       - "Moderately Feasible": Notable challenges with standard mitigation approaches
       - "Highly Feasible": Minimal challenges compared to typical Mercer Island properties

    5. Provide detailed recommendations that are:
       - Specific and actionable
       - Ordered by implementation sequence
       - Referenced to applicable codes where relevant
       - Include cost impact indicators (low/moderate/high)

    Return a JSON object with the following structure:
    - "location_analysis": Object containing "summary" and "recommendations" from the location analysis
    - "slope_analysis": Object containing "summary" and "recommendations" from the slope analysis
    - "overall_feasibility": Clear assessment using one of the four classifications defined above
    - "detailed_recommendations": List of 5-7 specific, actionable recommendations
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

    You are now responding to a direct question from a property owner about their feasibility report for a property on Mercer Island, WA. Your goal is to provide clear, accurate information that helps them understand the technical aspects of the report.

    When responding:
    1. Answer directly and clearly, using plain language whenever possible
    2. If technical terms are necessary, briefly define them in parentheses
    3. Cite specific data from the report to support your answers
    4. If the question goes beyond the scope of the report, acknowledge this and suggest what additional information might be needed
    5. Remain factual and objective rather than optimistic or pessimistic about development prospects
    6. For questions about regulations, reference specific codes (e.g., "According to Mercer Island Municipal Code 19.07.120...")
    7. For cost-related questions, provide ranges rather than specific figures and clarify that exact costs require contractor bids

    Feasibility Report Details:
    {json.dumps(report.dict(), indent=2)}

    User Question: "{user_query}"

    Provide a helpful, informative response that directly addresses the user's specific question. Use a professional but conversational tone appropriate for a property owner who may not have technical engineering background.
    """
    try:
        response = model.generate_content(prompt)
        logging.info(f"Raw chat response: {response.text[:100]}...")
        return response.text
    except Exception as e:
        logging.error(f"Error during chat with report: {str(e)}")
        return None