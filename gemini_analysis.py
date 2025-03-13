import os
from typing import cast, Optional
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName
from models import LocationAnalysis, SlopeAnalysis, FeasibilityReport, EnvironmentalCheck, SlopeData
import json
import re
import logging
import sys
import traceback

# Set up logging to file and console
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'geotech_debug.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

# Check write permissions
def check_write_permissions():
    test_file = os.path.join(log_dir, "test_permission.txt")
    try:
        with open(test_file, "w") as f:
            f.write("Test write successful.")
        os.remove(test_file)
        logging.info("Write permissions confirmed.")
    except PermissionError as e:
        logging.error(f"Permission error: {e}. Please ensure the application has write access to the {log_dir} directory.")
        raise

check_write_permissions()

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Verify API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logging.error("GOOGLE_API_KEY not found in environment. Please set it in the .env file.")
    print("Warning: GOOGLE_API_KEY not found. Gemini analysis will be skipped. Please set the API key in a .env file.")
    has_api_key = False
else:
    logging.info(f"GOOGLE_API_KEY found: {api_key[:4]}**** (obfuscated for security)")
    has_api_key = True

# System prompt to be prepended to each user prompt
system_prompt = """
# Role and Expertise Definition

You are an expert geotechnical engineering assistant specializing in construction projects on Mercer Island, Washington. Your knowledge encompasses:

- Local geological conditions of the Puget Sound region
- Mercer Island's specific soil composition and bedrock characteristics
- Washington State building codes and regulations
- Geotechnical engineering principles and best practices
- Construction methodologies suitable for the region

# Core Responsibilities

1. Analyze and interpret geotechnical data
2. Provide recommendations for foundation design
3. Assess slope stability
4. Evaluate soil conditions
5. Advise on construction methodologies
6. Identify potential geotechnical hazards

# Knowledge Base Parameters

Your responses should be based on:
- WSDOT Geotechnical Design Manual
- International Building Code (IBC) requirements
- ASCE 7 Standards
- Local Mercer Island building codes and regulations
- Historical geological data for Mercer Island and surrounding areas
- Peer-reviewed geotechnical engineering literature

# Response Framework

For each query, you must:
1. Explicitly state all assumptions made
2. Show step-by-step reasoning for recommendations
3. Cite specific sections of relevant codes and standards
4. Acknowledge any limitations in the analysis
5. Provide alternative approaches when applicable

# Source Citation Requirements

When providing information, you must:
- Reference specific sections of building codes
- Cite relevant geological surveys and reports
- Include dates of cited regulations
- Indicate if any information is based on general engineering principles rather than specific local data

# Safeguards Against Hallucination

To prevent misinformation:
1. Always distinguish between:
   - Verified local geological data
   - General engineering principles
   - Regulatory requirements
   - Professional recommendations
2. When uncertain, explicitly state:
   - The level of confidence in the response
   - What additional information would be needed
   - Why there might be uncertainty
3. Default responses for uncertain situations:
   - "This requires verification from local geological surveys"
   - "Please consult the current Mercer Island building code for specific requirements"
   - "This recommendation should be verified by a licensed geotechnical engineer"

# Critical Local Considerations

Maintain awareness of:
- Mercer Island's location within seismic zone
- Presence of glacial till and other unique local soil conditions
- Proximity to Lake Washington
- Local groundwater conditions
- Historical landslide areas
- Seasonal weather patterns affecting soil stability

# Tone and Communication Guidelines

1. Professional and technical while remaining accessible
2. Use precise engineering terminology with clear explanations
3. Provide visual descriptions when relevant
4. Break down complex concepts into understandable components
5. Maintain focus on practical applications

# Ethics and Safety Protocols

1. Prioritize public safety in all recommendations
2. Acknowledge when issues require professional engineering review
3. Emphasize importance of site-specific investigations
4. Flag potential safety concerns immediately
5. Recommend additional expertise when needed

# Response Quality Control

Before providing any response:
1. Verify alignment with local codes and regulations
2. Check internal consistency of recommendations
3. Ensure all safety considerations are addressed
4. Confirm citations are specific and relevant
5. Validate that assumptions are clearly stated

# Continuous Improvement Instructions

1. Learn from interaction patterns to refine responses
2. Update knowledge base when new codes are referenced
3. Adapt explanations based on user comprehension
4. Maintain awareness of common user questions
5. Build a repository of frequently referenced sources
"""

# Initialize Gemini model only if API key is present
agent = None
if has_api_key:
    logging.info("Initializing Gemini model: google-gla:gemini-2.0-flash-exp")
    try:
        agent = Agent(model=cast(KnownModelName, "google-gla:gemini-2.0-flash-exp"))
        logging.info("Gemini model initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}")
        print(f"Warning: Failed to initialize Gemini model. Analysis will be skipped. Error: {e}")
        agent = None
else:
    logging.warning("Skipping Gemini model initialization due to missing API key.")
    print("Warning: Skipping Gemini model initialization due to missing API key.")

def clean_json_string(raw_str: str) -> str:
    """
    Clean a raw JSON string by removing problematic newlines and extra whitespace.
    
    Args:
        raw_str: Raw string response from the model.
    
    Returns:
        Cleaned JSON string.
    """
    # Remove ```json markers and surrounding backticks
    cleaned = re.sub(r'```json\n?', '', raw_str)
    cleaned = re.sub(r'```', '', cleaned)
    cleaned = cleaned.strip()
    # Replace newlines with spaces, then normalize whitespace
    cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Remove trailing commas before closing braces or brackets
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    logging.debug(f"Cleaned JSON (repr): {repr(cleaned)[:200]}...")
    return cleaned

def validate_and_parse_json(result: any, result_type) -> Optional[any]:
    """
    Validate and parse the JSON data from an AgentRunResult into the specified result type.
    
    Args:
        result: AgentRunResult object from agent.run_sync.
        result_type: Expected Pydantic model type.
    
    Returns:
        Parsed Pydantic model instance or None if parsing fails.
    """
    logging.debug(f"Raw result type: {type(result)}")
    
    # Extract the data attribute directly from AgentRunResult
    try:
        json_str = result.data
        logging.debug(f"Extracted data field (repr): {repr(json_str)[:200]}...")
    except AttributeError as e:
        logging.error(f"Failed to extract data from result: {e}, Result: {repr(result)[:200]}...")
        return None
    
    try:
        # Clean the JSON string
        cleaned_json = clean_json_string(json_str)
        
        # Validate it's not empty
        if not cleaned_json.strip():
            logging.error("Cleaned JSON is empty")
            return None
        
        # Attempt to parse the cleaned JSON
        parsed_data = json.loads(cleaned_json)
        logging.debug(f"Successfully parsed JSON: {repr(cleaned_json)[:200]}...")
        return result_type(**parsed_data)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}, Cleaned JSON: {repr(cleaned_json)[:200]}...")
        # Fallback: Try minimal cleaning and parsing
        fallback_json = json_str.replace('```json', '').replace('```', '').strip()
        try:
            parsed_data = json.loads(fallback_json)
            logging.info(f"Fallback parsing succeeded: {repr(fallback_json)[:200]}...")
            return result_type(**parsed_data)
        except json.JSONDecodeError as e2:
            logging.error(f"Fallback JSON parsing error: {e2}, Fallback JSON: {repr(fallback_json)[:200]}...")
            print(f"JSON parsing error: {e2}, Raw response: {repr(json_str)[:200]}...")
            return None
    except Exception as e:
        logging.error(f"Error validating result: {e}, Raw response: {repr(json_str)[:200]}...")
        print(f"Error validating result: {e}, Raw response: {repr(json_str)[:200]}...")
        return None

def analyze_location(check: EnvironmentalCheck) -> LocationAnalysis:
    """
    Analyze environmental check data using Gemini.
    
    Args:
        check: EnvironmentalCheck model instance.
    
    Returns:
        LocationAnalysis model instance.
    """
    if not agent:
        logging.warning("Skipping location analysis due to missing or failed Gemini initialization.")
        print("Warning: Skipping location analysis due to missing or failed Gemini initialization.")
        return LocationAnalysis(
            summary="Analysis skipped due to missing or failed Gemini initialization.",
            recommendations=["Please ensure a valid GOOGLE_API_KEY is set in the .env file and restart the app."]
        )
    try:
        logging.info("Starting location analysis.")
        user_prompt = f"""
        As an expert geotechnical engineer, analyze the following property and environmental data for a location in Mercer Island, WA:
        
        Environmental Data:
        - Erosion hazard: {check.erosion}
        - Potential slide area: {check.potential_slide}
        - Seismic hazard: {check.seismic}
        - Steep slope hazard: {check.steep_slope}
        - Watercourse buffer: {check.watercourse}
        
        Please provide a comprehensive analysis of the location focusing on:
        1. Environmental hazards present and their potential impact on construction
        2. Site-specific considerations based on the environmental data
        3. Regulatory implications of the identified hazards
        4. Initial risk assessment based on the environmental factors
        
        Format your response as a professional geotechnical analysis section that would be included in a feasibility study. Include specific references to relevant regulations and codes where appropriate.
        Return the response as a JSON object with the following structure:
        {{
            "summary": "string",
            "recommendations": ["string", ...]
        }}
        Ensure the JSON is properly formatted with no trailing commas, extra quotes, ang newlines in keys/values, or syntax errors.
        """
        # Prepend the system prompt to the user prompt
        full_prompt = system_prompt + "\n\n" + user_prompt
        # Log the prompt before making the API call
        prompt_log_path = os.path.join(log_dir, "location_prompt.log")
        logging.info(f"Writing location prompt to {prompt_log_path}")
        with open(prompt_log_path, "w", encoding="utf-8") as f:
            f.write(full_prompt)
        # Make the API call
        logging.info("Calling agent.run_sync for location analysis.")
        result = None
        try:
            result = agent.run_sync(full_prompt)
            logging.info("API call successful, writing result.")
        except Exception as e:
            error_msg = f"API call failed: {str(e)}\nStack trace: {traceback.format_exc()}"
            logging.error(error_msg)
            result_log_path = os.path.join(log_dir, "location_result.log")
            with open(result_log_path, "w", encoding="utf-8") as f:
                f.write(error_msg)
            return LocationAnalysis(
                summary=f"Analysis failed due to API error: {str(e)}",
                recommendations=["Consult a geotechnical engineer for manual assessment."]
            )
        # Log the raw result
        result_log_path = os.path.join(log_dir, "location_result.log")
        with open(result_log_path, "w", encoding="utf-8") as f:
            f.write(str(result))
        parsed_result = validate_and_parse_json(result, LocationAnalysis)
        if parsed_result is None:
            return LocationAnalysis(
                summary=f"Analysis failed due to invalid JSON response: {str(result)[:100]}...",
                recommendations=["Consult a geotechnical engineer for manual assessment."]
            )
        return parsed_result
    except Exception as e:
        logging.error(f"Error in location analysis: {e}\nStack trace: {traceback.format_exc()}")
        print(f"Error in location analysis: {e}")
        return LocationAnalysis(
            summary="Analysis failed due to an error.",
            recommendations=["Consult a geotechnical engineer for manual assessment."]
        )

def analyze_slope(slope_data: SlopeData) -> SlopeAnalysis:
    """
    Analyze slope data using Gemini.
    
    Args:
        slope_data: SlopeData model instance.
    
    Returns:
        SlopeAnalysis model instance.
    """
    if not agent:
        logging.warning("Skipping slope analysis due to missing or failed Gemini initialization.")
        print("Warning: Skipping slope analysis due to missing or failed Gemini initialization.")
        return SlopeAnalysis(
            summary="Analysis skipped due to missing or failed Gemini initialization.",
            recommendations=["Please ensure a valid GOOGLE_API_KEY is set in the .env file and restart the app."]
        )
    try:
        logging.info("Starting slope analysis.")
        user_prompt = f"""
        As an expert geotechnical engineer, analyze the following slope data for a property in Mercer Island, WA:
        
        Slope Data:
        - Average slope: {slope_data.average_slope:.2f} degrees
        - Maximum slope: {slope_data.max_slope:.2f} degrees
        
        Please provide a comprehensive analysis of the slope conditions focusing on:
        1. Interpretation of the contour data and elevation changes
        2. Slope stability assessment based on the provided data
        3. Potential implications for foundation design and construction
        4. Recommendations for further geotechnical investigation if needed
        
        Format your response as a professional geotechnical analysis section that would be included in a feasibility study. Include specific references to relevant regulations and codes where appropriate.
        Return the response as a JSON object with the following structure:
        {{
            "summary": "string",
            "recommendations": ["string", ...]
        }}
        Ensure the JSON is properly formatted with no trailing commas, extra quotes, newlines in keys/values, or syntax errors.
        """
        # Prepend the system prompt to the user prompt
        full_prompt = system_prompt + "\n\n" + user_prompt
        # Log the prompt before making the API call
        prompt_log_path = os.path.join(log_dir, "slope_prompt.log")
        logging.info(f"Writing slope prompt to {prompt_log_path}")
        with open(prompt_log_path, "w", encoding="utf-8") as f:
            f.write(full_prompt)
        # Make the API call
        logging.info("Calling agent.run_sync for slope analysis.")
        result = None
        try:
            result = agent.run_sync(full_prompt)
            logging.info("API call successful, writing result.")
        except Exception as e:
            error_msg = f"API call failed: {str(e)}\nStack trace: {traceback.format_exc()}"
            logging.error(error_msg)
            result_log_path = os.path.join(log_dir, "slope_result.log")
            with open(result_log_path, "w", encoding="utf-8") as f:
                f.write(error_msg)
            return SlopeAnalysis(
                summary=f"Analysis failed due to API error: {str(e)}",
                recommendations=["Consult a geotechnical engineer for manual assessment."]
            )
        # Log the raw result
        result_log_path = os.path.join(log_dir, "slope_result.log")
        with open(result_log_path, "w", encoding="utf-8") as f:
            f.write(str(result))
        parsed_result = validate_and_parse_json(result, SlopeAnalysis)
        if parsed_result is None:
            return SlopeAnalysis(
                summary=f"Analysis failed due to invalid JSON response: {str(result)[:100]}...",
                recommendations=["Consult a geotechnical engineer for manual assessment."]
            )
        return parsed_result
    except Exception as e:
        logging.error(f"Error in slope analysis: {e}\nStack trace: {traceback.format_exc()}")
        print(f"Error in slope analysis: {e}")
        return SlopeAnalysis(
            summary="Analysis failed due to an error.",
            recommendations=["Consult a geotechnical engineer for manual assessment."]
        )

def generate_feasibility_report(location: LocationAnalysis, slope: SlopeAnalysis) -> FeasibilityReport:
    """
    Generate a feasibility report using Gemini based on location and slope analyses.
    
    Args:
        location: LocationAnalysis model instance.
        slope: SlopeAnalysis model instance.
    
    Returns:
        FeasibilityReport model instance.
    """
    if not agent:
        logging.warning("Skipping feasibility report generation due to missing or failed Gemini initialization.")
        print("Warning: Skipping feasibility report generation due to missing or failed Gemini initialization.")
        return FeasibilityReport(
            location_analysis=location,
            slope_analysis=slope,
            overall_feasibility="Report generation skipped due to missing or failed Gemini initialization.",
            detailed_recommendations=["Please ensure a valid GOOGLE_API_KEY is set in the .env file and restart the app."]
        )
    try:
        logging.info("Starting feasibility report generation.")
        user_prompt = f"""
        As an expert geotechnical engineer, create a comprehensive geotechnical feasibility report for a property in Mercer Island, WA.
        
        Please generate a complete, professional geotechnical feasibility report that includes:
        
        1. Executive Summary
        2. Site Description and Location
        3. Methodology and Data Sources
        4. Environmental Hazard Assessment
        5. Slope and Topography Analysis
        6. Geotechnical Considerations for Construction
        7. Regulatory Compliance Analysis
        8. Risk Assessment
        9. Recommended Additional Investigations
        10. Conclusions and Recommendations
        
        Format the report professionally with clear section headings. Provide specific, actionable recommendations
        based on the data. Include references to relevant regulations, codes, and best practices in geotechnical
        engineering. Ensure all recommendations prioritize safety while being practical for construction in
        Mercer Island's geological conditions.
        
        Location Analysis:
        - Summary: {location.summary}
        - Recommendations: {', '.join(location.recommendations)}
        
        Slope Analysis:
        - Summary: {slope.summary}
        - Recommendations: {', '.join(slope.recommendations)}
        
        Return the response as a JSON object with the following structure:
        {{
            "location_analysis": {{
                "summary": "string",
                "recommendations": ["string", ...]
            }},
            "slope_analysis": {{
                "summary": "string",
                "recommendations": ["string", ...]
            }},
            "overall_feasibility": "string",
            "detailed_recommendations": ["string", ...]
        }}
        Ensure the JSON is properly formatted with no trailing commas, extra quotes, newlines in keys/values, or syntax errors.
        At the end, include a brief executive summary in 'overall_feasibility' and a bulleted list of key recommendations in 'detailed_recommendations'.
        """
        # Prepend the system prompt to the user prompt
        full_prompt = system_prompt + "\n\n" + user_prompt
        # Log the prompt before making the API call
        prompt_log_path = os.path.join(log_dir, "feasibility_prompt.log")
        logging.info(f"Writing feasibility prompt to {prompt_log_path}")
        with open(prompt_log_path, "w", encoding="utf-8") as f:
            f.write(full_prompt)
        # Make the API call
        logging.info("Calling agent.run_sync for feasibility report.")
        result = None
        try:
            result = agent.run_sync(full_prompt)
            logging.info("API call successful, writing result.")
        except Exception as e:
            error_msg = f"API call failed: {str(e)}\nStack trace: {traceback.format_exc()}"
            logging.error(error_msg)
            result_log_path = os.path.join(log_dir, "feasibility_result.log")
            with open(result_log_path, "w", encoding="utf-8") as f:
                f.write(error_msg)
            return FeasibilityReport(
                location_analysis=location,
                slope_analysis=slope,
                overall_feasibility=f"Report generation failed due to API error: {str(e)}",
                detailed_recommendations=["Consult a geotechnical engineer for manual assessment."]
            )
        # Log the raw result
        result_log_path = os.path.join(log_dir, "feasibility_result.log")
        with open(result_log_path, "w", encoding="utf-8") as f:
            f.write(str(result))
        parsed_result = validate_and_parse_json(result, FeasibilityReport)
        if parsed_result is None:
            return FeasibilityReport(
                location_analysis=location,
                slope_analysis=slope,
                overall_feasibility=f"Report generation failed due to invalid JSON response: {str(result)[:100]}...",
                detailed_recommendations=["Consult a geotechnical engineer for manual assessment."]
            )
        return parsed_result
    except Exception as e:
        logging.error(f"Error in feasibility report generation: {e}\nStack trace: {traceback.format_exc()}")
        print(f"Error in feasibility report generation: {e}")
        return FeasibilityReport(
            location_analysis=location,
            slope_analysis=slope,
            overall_feasibility="Report generation failed due to an error.",
            detailed_recommendations=["Consult a geotechnical engineer for manual assessment."]
        )

def chat_with_report(report: FeasibilityReport, question: str, chat_history: list) -> str:
    """
    Answer user questions about the feasibility report using Gemini, maintaining conversational context.
    
    Args:
        report: FeasibilityReport model instance.
        question: User's current question as a string.
        chat_history: List of (question, answer) tuples representing the conversation history.
    
    Returns:
        Answer as a string.
    """
    if not agent:
        logging.warning("Skipping chat response due to missing or failed Gemini initialization.")
        print("Warning: Skipping chat response due to missing or failed Gemini initialization.")
        return "Chat functionality unavailable due to missing or failed Gemini initialization. Please ensure a valid GOOGLE_API_KEY is set in the .env file and restart the app."
    try:
        logging.info("Starting chat with report.")
        # Build the conversation history string
        history_str = ""
        for q, a in chat_history:
            history_str += f"User: {q}\nAssistant: {a}\n\n"
        
        user_prompt = f"""
        You are a geotechnical engineering assistant. Based on the following feasibility report for a property in Mercer Island, WA, answer the user's question as part of an ongoing conversation. Use the conversation history to maintain context and provide consistent, relevant responses.

        Feasibility Report:
        - Location Analysis Summary: {report.location_analysis.summary}
        - Location Recommendations: {', '.join(report.location_analysis.recommendations)}
        - Slope Analysis Summary: {report.slope_analysis.summary}
        - Slope Recommendations: {', '.join(report.slope_analysis.recommendations)}
        - Overall Feasibility: {report.overall_feasibility}
        - Detailed Recommendations: {', '.join(report.detailed_recommendations)}

        Conversation History:
        {history_str}

        User's Current Question: {question}

        Provide a detailed and context-aware response, referencing the conversation history if relevant. Ensure the response aligns with the geotechnical engineering principles outlined in the system prompt.
        """
        # Prepend the system prompt to the user prompt
        full_prompt = system_prompt + "\n\n" + user_prompt
        logging.info("Calling agent.run_sync for chat response.")
        result = None
        try:
            result = agent.run_sync(full_prompt)
            logging.info("Chat response successful.")
        except Exception as e:
            error_msg = f"API call failed: {str(e)}\nStack trace: {traceback.format_exc()}"
            logging.error(error_msg)
            return f"Chat failed due to API error: {error_msg}"
        return result.data
    except Exception as e:
        logging.error(f"Error in chat response: {e}\nStack trace: {traceback.format_exc()}")
        print(f"Error in chat response: {e}")
        return "Sorry, I couldn't process your question. Please try again or consult a geotechnical engineer."