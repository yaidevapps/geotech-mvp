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

# System prompt for the Gemini model
SYSTEM_PROMPT = """
You are a geotechnical engineering assistant specializing in property feasibility analysis for Mercer Island, WA. Your role is to analyze location data, slope data, and environmental hazards to provide detailed feasibility reports for construction projects. Use the provided data to give accurate, professional, and concise responses. Return all responses as a JSON object with the required fields, enclosed in triple backticks (```json ... ```). If any data is missing or unclear, state your assumptions clearly and provide default values where necessary.
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
    """Analyze the location using Gemini."""
    if model is None:
        logging.warning("Skipping location analysis due to missing or failed Gemini initialization.")
        return None

    prompt = f"{SYSTEM_PROMPT}\n\nAnalyze the location at latitude {latitude}, longitude {longitude}, with address '{address}' on Mercer Island, WA. Provide a detailed analysis including proximity to amenities, zoning information, and potential geotechnical concerns. Return a JSON object with 'summary' (string) and 'recommendations' (list of strings), enclosed in triple backticks (```json ... ```)."
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
    """Analyze the slope using Gemini."""
    if model is None:
        logging.warning("Skipping slope analysis due to missing or failed Gemini initialization.")
        return None

    prompt = f"{SYSTEM_PROMPT}\n\nAnalyze a slope of {slope:.2f}% with an elevation difference of {elevation_diff:.2f} feet over a distance of {distance:.2f} meters on Mercer Island, WA. Discuss stability, potential for erosion, and construction challenges. Return a JSON object with 'summary' (string) and 'recommendations' (list of strings), enclosed in triple backticks (```json ... ```)."
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
    """Generate a feasibility report using Gemini."""
    if model is None:
        logging.warning("Skipping feasibility report generation due to missing or failed Gemini initialization.")
        return None

    prompt = f"{SYSTEM_PROMPT}\n\nGenerate a feasibility report for a property at '{address}' on Mercer Island, WA.\n"
    if location_analysis:
        prompt += f"Location Analysis: {location_analysis.dict()}\n"
    if slope_analysis:
        prompt += f"Slope Analysis: {slope_analysis.dict()}\n"
    prompt += f"Environmental Hazards: {environmental_hazards}\n"
    prompt += "Provide a detailed feasibility report for construction, including recommendations and risk assessment. Return a JSON object with 'location_analysis' (object with 'summary' and 'recommendations'), 'slope_analysis' (object with 'summary' and 'recommendations'), 'overall_feasibility' (string), and 'detailed_recommendations' (list of strings), enclosed in triple backticks (```json ... ```)."
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
    """Chat with the user about the feasibility report using Gemini."""
    if model is None:
        logging.warning("Skipping chat due to missing or failed Gemini initialization.")
        return None

    prompt = f"{SYSTEM_PROMPT}\n\nFeasibility Report: {report.dict()}\nUser Query: {user_query}\nProvide a detailed and professional response to the user's query based on the feasibility report."
    try:
        response = model.generate_content(prompt)
        logging.info(f"Raw chat response: {response.text[:100]}...")
        return response.text
    except Exception as e:
        logging.error(f"Error during chat with report: {str(e)}")
        return None