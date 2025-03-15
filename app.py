import streamlit as st
import numpy as np
import logging
import json
import re
import geopandas as gpd
from shapely.geometry import shape

from models import Address, Coordinates, Property, SlopeData, EnvironmentalCheck, FeasibilityReport, SlopeAnalysis
from geo_processing import geocode_address, extract_property, calculate_slope, check_environmental_hazards, create_map, HAZARD_FILES
from gemini_analysis import analyze_location, analyze_slope, generate_feasibility_report, chat_with_report

# Configure logging to ensure INFO level logs are visible
logging.basicConfig(level=logging.INFO)

# Define GeoJSON files for map display
GEOJSON_FILES = {
    "Property Lines": "data/Mercer_Island_Basemap_Data_Layers_PropertyLine.geojson",
    "Contours": "data/Mercer_Island_Environmental_Layers_10ftLidarContours.geojson",
    "Erosion Hazard": "data/Mercer_Island_Environmental_Layers_Erosion.geojson",
    "Potential Slide Hazard": "data/Mercer_Island_Environmental_Layers_PotentialSlideAreas.geojson",
    "Seismic Hazard": "data/Mercer_Island_Environmental_Layers_Seismic.geojson",
    "Steep Slope Hazard": "data/Mercer_Island_Environmental_Layers_SteepSlope.geojson",
    "Watercourse Buffer": "data/Mercer_Island_Environmental_Layers_WatercourseBufferSetback.geojson",
}

def log_feedback(user_input: str, model_output: str, feedback: str):
    """Log user feedback to a file for review."""
    with open("feedback_log.txt", "a") as f:
        f.write(f"Input: {user_input}\nOutput: {model_output}\nFeedback: {feedback}\n\n")

def perform_analysis(street: str, zip_code: str) -> None:
    """Perform the full property analysis and store results in session state."""
    address_str = f"{street}, Mercer Island, WA"
    if zip_code:
        address_str += f", {zip_code}"
    address = Address(street=street, zip_code=zip_code if zip_code else None)
    coordinates = geocode_address(address)

    if not coordinates:
        st.error("Failed to geocode address. Please check the input and try again.")
        return

    st.session_state.coordinates = coordinates
    st.session_state.address = address_str
    st.success(f"Geocoded Address: ({coordinates.latitude}, {coordinates.longitude})")

    property_data = extract_property(coordinates)
    if not property_data:
        st.error("No property found at the given coordinates.")
        return
    st.session_state.property = property_data

    slope_data = calculate_slope(property_data)
    if not slope_data:
        st.error("Failed to calculate slope data.")
        return
    logging.info(
        f"Slope Data - Average: {slope_data.average_slope}, Max: {slope_data.max_slope}, Distance: {slope_data.average_distance}"
    )
    st.session_state.slope_data = slope_data

    environmental_check = check_environmental_hazards(property_data)
    if not environmental_check:
        st.error("Failed to check environmental hazards.")
        return
    st.session_state.environmental_check = environmental_check

    # Calculate lake proximity for consistency
    property_geom = shape(property_data.geometry)
    property_geom_proj = gpd.GeoSeries([property_geom], crs="EPSG:4326").to_crs("EPSG:32610")[0]
    erosion_gdf = gpd.read_file(HAZARD_FILES["erosion"]).to_crs("EPSG:32610")
    lake_proximity = erosion_gdf.intersects(property_geom_proj.buffer(100)).any()

    with st.spinner("Performing location analysis..."):
        location_analysis = analyze_location(
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
            address=address_str,
            hazards=environmental_check.model_dump(),
            lake_proximity=lake_proximity,
        )
    st.session_state.location_analysis = location_analysis

    with st.spinner("Performing slope analysis..."):
        elevation_diff = slope_data.average_distance * np.tan(np.radians(slope_data.average_slope))
        slope_analysis = analyze_slope(
            slope=slope_data.average_slope,
            elevation_diff=elevation_diff,
            distance=slope_data.average_distance,
            lake_proximity=lake_proximity,
        )
        if slope_analysis:
            st.session_state.slope_analysis = slope_analysis
        else:
            st.session_state.slope_analysis = SlopeAnalysis(
                summary="Slope analysis failed due to processing error.",
                recommendations=[],
                verification_needed=["Slope data processing"],
            )

    with st.spinner("Generating feasibility report..."):
        environmental_hazards = environmental_check.model_dump()
        feasibility_report = generate_feasibility_report(
            address=address_str,
            slope_analysis=slope_analysis,
            location_analysis=location_analysis,
            environmental_hazards=environmental_hazards,
            slope_data=slope_data,
            lake_proximity=lake_proximity,
        )
    st.session_state.feasibility_report = feasibility_report

def display_report():
    """Display the feasibility report with styled sections, feedback buttons, and verification needs."""
    required_keys = ["coordinates", "property", "feasibility_report"]
    if not all(key in st.session_state and st.session_state[key] is not None for key in required_keys):
        st.info("No analysis results available yet. Please enter an address and click 'Analyze Property'.")
        return

    report = st.session_state.feasibility_report
    if not (hasattr(report, "location_analysis") and hasattr(report, "slope_analysis")):
        st.warning("Feasibility report is incomplete. Please re-run the analysis.")
        return

    # Inject custom CSS with styles for chat section
    st.markdown(
        """
    <style>
    .feasibility-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 12px;
    }
    .hazard-present {
        background-color: #fef2f2;
        border: 1px solid #f87171;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .hazard-absent {
        background-color: #ecfdf5;
        border: 1px solid #34d399;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .status-dot-present {
        width: 12px;
        height: 12px;
        background-color: #ef4444;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-dot-absent {
        width: 12px;
        height: 12px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .critical-card {
        border: 1px solid #f87171;
        background-color: #fee2e2;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 8px;
    }
    .major-card {
        border: 1px solid #fb923c;
        background-color: #ffedd5;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 8px;
    }
    .minor-card {
        border: 1px solid #34d399;
        background-color: #ecfdf5;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 8px;
    }
    .priority-label {
        font-weight: bold;
    }
    .confidence-level {
        font-weight: bold;
    }
    div.stButton > button {
        background-color: #e5e7eb;
        color: #374151;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        border: none;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #f4a8a8;  /* Light red on hover */
        color: #374151;  /* Ensure text color stays the same */
    }
    /* Chat section styles */
    .chat-subheader {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 12px;
    }
    .user-message {
        background-color: #f3f4f6;  /* Light gray */
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 8px;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .assistant-message {
        background-color: #e0f2fe;  /* Light blue */
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 8px;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .chat-recommendation {
        border: 1px solid #d1d5db;
        background-color: #f9fafb;
        padding: 8px;
        border-radius: 8px;
        margin-bottom: 6px;
    }
    .chat-field-label {
        font-weight: bold;
        color: #1e40af;  /* Blue for labels */
        margin-right: 4px;
    }
    .chat-disclaimer {
        font-size: 12px;
        font-style: italic;
        color: #6b7280;  /* Muted gray */
        margin-top: 8px;
    }
    @media (max-width: 640px) {
        .recommendation-row {
            display: block;
        }
        .recommendation-text, div.stButton > button {
            width: 100%;
            margin-bottom: 8px;
        }
        .user-message, .assistant-message {
            max-width: 100%;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.subheader("Feasibility Report", divider="gray")

    with st.expander("Property Map", expanded=True):
        with st.spinner("Loading map..."):
            map_container = st.container()
            with map_container:
                st.markdown(
                    """
                    <style>
                    .map-container {
                        max-width: 100%;
                        overflow-x: auto;
                    }
                    .folium-map {
                        max-width: 100% !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                create_map(st.session_state.coordinates, st.session_state.property, GEOJSON_FILES)

    # Display Overall Feasibility with UI Feedback
    st.markdown(
        f'<div class="feasibility-title">Overall Feasibility: {report.overall_feasibility}</div>',
        unsafe_allow_html=True,
    )

    # Add UI feedback for potential feasibility mismatch
    try:
        if (
            report.overall_feasibility.startswith("Marginally")
            and hasattr(report, "slope_analysis")
            and report.slope_analysis
            and "flat" in report.slope_analysis.summary.lower()
            and hasattr(report, "hazard_layers")
            and report.hazard_layers
            and not any("Present" in h.split(":")[1].strip() for h in report.hazard_layers)
        ):
            st.warning(
                "The 'Marginally Feasible' rating may be overly conservative given the flat slope and absence of hazards. "
                "Consider verifying soil bearing capacity to confirm a higher feasibility rating (e.g., 'Highly Feasible')."
            )
    except AttributeError as e:
        logging.error(f"Error checking feasibility mismatch: {e}")

    with st.expander("Hazard Layer Information", expanded=True):
        if hasattr(report, "hazard_layers"):
            for hazard in report.hazard_layers:
                is_present = "Present" in hazard and "Not Present" not in hazard
                logging.info(f"Hazard: {hazard}, is_present: {is_present}")
                st.markdown(
                    f'<div class="{"hazard-present" if is_present else "hazard-absent"}">'
                    f'<span class="{"status-dot-present" if is_present else "status-dot-absent"}"></span>'
                    f"{hazard}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.write("Hazard layer information unavailable.")

    with st.expander("Location Analysis"):
        st.write(
            "**Summary:**",
            report.location_analysis.summary if report.location_analysis else "Analysis unavailable",
        )
        if report.location_analysis and hasattr(report.location_analysis, "recommendations"):
            st.write("**Recommendations:**")
            for rec in report.location_analysis.recommendations:
                priority = rec.split("]:")[0][1:].lower()
                card_class = f"{priority}-card"
                parts = rec.split(" - Confidence: ")
                if len(parts) == 2:
                    body = parts[0]
                    confidence = "Confidence: " + parts[1]
                    priority_part, body_part = body.split("]: ", 1)
                    styled_rec = (
                        f'<span class="priority-label">{priority_part}]:</span> '
                        f'{body_part} - <span class="confidence-level">{confidence}</span>'
                    )
                else:
                    styled_rec = rec
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f'<div class="{card_class}">{styled_rec}</div>', unsafe_allow_html=True)
                with col2:
                    if st.button("Flag This", key=f"loc_{rec[:50]}"):
                        log_feedback("Location Analysis", rec, "User flagged as inconsistent")
                        st.success("Feedback logged.")

    with st.expander("Slope Analysis"):
        if report.slope_analysis and isinstance(report.slope_analysis, SlopeAnalysis):
            st.write("**Summary:**", report.slope_analysis.summary)
            if report.slope_analysis.recommendations:
                st.write("**Recommendations:**")
                for rec in report.slope_analysis.recommendations:
                    priority = rec.split("]:")[0][1:].lower()
                    card_class = f"{priority}-card"
                    parts = rec.split(" - Confidence: ")
                    if len(parts) == 2:
                        body = parts[0]
                        confidence = "Confidence: " + parts[1]
                        priority_part, body_part = body.split("]: ", 1)
                        styled_rec = (
                            f'<span class="priority-label">{priority_part}]:</span> '
                            f'{body_part} - <span class="confidence-level">{confidence}</span>'
                        )
                    else:
                        styled_rec = rec
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f'<div class="{card_class}">{styled_rec}</div>', unsafe_allow_html=True)
                    with col2:
                        if st.button("Flag This", key=f"slope_{rec[:50]}"):
                            log_feedback("Slope Analysis", rec, "User flagged as inconsistent")
                            st.success("Feedback logged.")
        else:
            st.write("**Summary:**", "Slope analysis unavailable or failed.")

    with st.expander("Detailed Recommendations"):
        if hasattr(report, "detailed_recommendations"):
            for rec in report.detailed_recommendations:
                priority = rec.split("]:")[0][1:].lower()
                card_class = f"{priority}-card"
                parts = rec.split(" - Confidence: ")
                if len(parts) == 2:
                    body = parts[0]
                    confidence = "Confidence: " + parts[1]
                    priority_part, body_part = body.split("]: ", 1)
                    styled_rec = (
                        f'<span class="priority-label">{priority_part}]:</span> '
                        f'{body_part} - <span class="confidence-level">{confidence}</span>'
                    )
                else:
                    styled_rec = rec
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f'<div class="{card_class}">{styled_rec}</div>', unsafe_allow_html=True)
                with col2:
                    if st.button("Flag This", key=f"detail_{rec[:50]}"):
                        log_feedback("Detailed Recommendations", rec, "User flagged as inconsistent")
                        st.success("Feedback logged.")
        else:
            st.write("No detailed recommendations available.")

    with st.expander("Verification Needed"):
        if hasattr(report, "verification_needed") and report.verification_needed:
            st.write("**Additional Data Required:**")
            for item in report.verification_needed:
                st.markdown(f"- {item}")
        else:
            st.write("No additional verification required based on current data.")

def main():
    st.title("Geotechnical Engineering Assistant - Mercer Island, WA")

    # Initialize session state variables if they don’t exist
    session_state_vars = [
        "coordinates",
        "address",
        "property",
        "slope_data",
        "environmental_check",
        "location_analysis",
        "slope_analysis",
        "feasibility_report",
        "chat_history",
        "street_input",
        "zip_input",
    ]
    for var in session_state_vars:
        if var not in st.session_state:
            st.session_state[var] = None if var not in ["chat_history", "street_input", "zip_input"] else (
                [] if var == "chat_history" else ""
            )

    with st.sidebar:
        st.header("Address Input")
        st.markdown(
            """
        **How to Use This App:**
        1. Enter a Mercer Island street address (e.g., "1925 82nd Ave SE") and optional ZIP code.
        2. Click "Analyze Property" to generate a feasibility report with slope, hazard, and location insights.
        3. View results in the "Analysis" tab—expand sections for details and use "Flag This" to report issues.
        4. Ask questions about your report in the "Chat" tab.
        """
        )
        street = st.text_input(
            "Street Address",
            value=st.session_state.street_input,
            placeholder="e.g., 1234 Main St",
            key="street",
        )
        zip_code = st.text_input(
            "ZIP Code (optional)",
            value=st.session_state.zip_input,
            placeholder="e.g., 98040",
            key="zip",
        )
        analyze_button = st.button("Analyze Property")

        st.session_state.street_input = street
        st.session_state.zip_input = zip_code

        if analyze_button and street:
            perform_analysis(street, zip_code)

    analysis_tab, chat_tab = st.tabs(["Analysis", "Chat"])

    with analysis_tab:
        display_report()

    with chat_tab:
        if st.session_state.feasibility_report and hasattr(st.session_state.feasibility_report, "location_analysis"):
            st.markdown('<div class="chat-subheader">Chat with Your Feasibility Report</div>', unsafe_allow_html=True)

            # Add a button to clear chat history
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared.")

            chat_container = st.container()
            with chat_container:
                for idx, (question, answer) in enumerate(st.session_state.chat_history):
                    # User message
                    st.markdown(
                        f'<div class="user-message"><strong>You:</strong> {question}</div>', unsafe_allow_html=True
                    )

                    # Assistant message
                    logging.info(f"Raw chat response: {answer}")
                    try:
                        json_match = re.match(r"```json\s*(.*?)\s*```", answer, re.DOTALL)
                        if json_match:
                            main_response = json_match.group(1).strip()
                        else:
                            main_response = answer

                        disclaimer_pattern = r"(I’m limited to geotechnical analysis.*|If this response seems inaccurate, please flag it for review\.)"
                        disclaimer_match = re.search(disclaimer_pattern, main_response, re.DOTALL)
                        if disclaimer_match:
                            disclaimer = disclaimer_match.group(0).strip()
                            main_response = main_response[: disclaimer_match.start()].strip()
                        else:
                            disclaimer = ""

                        response_clean = main_response.replace("Assistant: ", "").strip()
                        response_dict = None
                        if response_clean.startswith("{") and response_clean.endswith("}"):
                            try:
                                response_dict = json.loads(response_clean)
                            except json.JSONDecodeError as e:
                                logging.warning(f"JSON parsing failed: {e}. Falling back to raw response.")
                                response_dict = None

                        st.markdown('<div class="assistant-message">', unsafe_allow_html=True)
                        st.markdown("<strong>Assistant:</strong>", unsafe_allow_html=True)

                        if response_dict:
                            if "geotechnical_analysis" in response_dict:
                                geo_analysis = response_dict["geotechnical_analysis"]
                                if "subject" in geo_analysis:
                                    st.markdown(
                                        f'<div><span class="chat-field-label">Subject:</span> {geo_analysis["subject"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if "analysis" in geo_analysis:
                                    st.markdown(
                                        f'<div style="margin-top: 8px;"><span class="chat-field-label">Analysis:</span> {geo_analysis["analysis"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if "liquefaction_assessment" in geo_analysis:
                                    st.markdown(
                                        f'<div style="margin-top: 8px;"><span class="chat-field-label">Liquefaction Assessment:</span> {geo_analysis["liquefaction_assessment"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if "seismic_design_codes" in geo_analysis:
                                    st.markdown(
                                        f'<div style="margin-top: 8px;"><span class="chat-field-label">Seismic Design Codes:</span> {geo_analysis["seismic_design_codes"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if "recommendations_summary" in geo_analysis:
                                    st.markdown(
                                        f'<div style="margin-top: 8px;"><span class="chat-field-label">Recommendations Summary:</span> {geo_analysis["recommendations_summary"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if "recommendations" in geo_analysis:
                                    st.markdown(
                                        '<div style="margin-top: 8px;"><span class="chat-field-label">Recommendations:</span></div>',
                                        unsafe_allow_html=True,
                                    )
                                    for rec in geo_analysis["recommendations"]:
                                        match = re.match(
                                            r"$$     (.*?)     $$:\s*(.*?)\s*$$     (.*?)     $$\s*-\s*Confidence:\s*(.*?)$", rec
                                        )
                                        if match:
                                            priority = f"[{match.group(1)}]"
                                            description = match.group(2).strip()
                                            cost = match.group(3).strip()
                                            confidence = match.group(4).strip()
                                        else:
                                            priority = "Unknown"
                                            description = rec
                                            cost = "Unknown"
                                            confidence = "Unknown"
                                        st.markdown(
                                            f'<div class="chat-recommendation">'
                                            f'<div><span class="chat-field-label">Description:</span> {description}</div>'
                                            f'<div><span class="chat-field-label">Priority:</span> <span class="priority-label">{priority}</span></div>'
                                            f'<div><span class="chat-field-label">Cost:</span> {cost}</div>'
                                            f'<div><span class="chat-field-label">Confidence:</span> <span class="confidence-level">{confidence}</span></div>'
                                            f'</div>',
                                            unsafe_allow_html=True,
                                        )
                                if "limitations" in geo_analysis:
                                    st.markdown(
                                        f'<div style="margin-top: 8px;"><span class="chat-field-label">Limitations:</span> {geo_analysis["limitations"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if "additional_information" in geo_analysis:
                                    st.markdown(
                                        f'<div style="margin-top: 8px;"><span class="chat-field-label">Additional Information:</span> {geo_analysis["additional_information"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                            elif "foundation_recommendations" in response_dict:
                                rec_data = response_dict["foundation_recommendations"]
                                if "summary" in rec_data:
                                    st.markdown(
                                        f'<div><span class="chat-field-label">Summary:</span> {rec_data["summary"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if "detailed_explanation" in rec_data:
                                    st.markdown(
                                        f'<div style="margin-top: 8px;"><span class="chat-field-label">Detailed Explanation:</span> {rec_data["detailed_explanation"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if "recommendations" in rec_data:
                                    st.markdown(
                                        '<div style="margin-top: 8px;"><span class="chat-field-label">Recommendations:</span></div>',
                                        unsafe_allow_html=True,
                                    )
                                    for rec in rec_data["recommendations"]:
                                        match = re.match(
                                            r"$$     (.*?)     $$:\s*(.*?)\s*$$     (.*?)     $$\s*-\s*Confidence:\s*(.*?)$", rec
                                        )
                                        if match:
                                            priority = f"[{match.group(1)}]"
                                            description = match.group(2).strip()
                                            cost = match.group(3).strip()
                                            confidence = match.group(4).strip()
                                        else:
                                            priority = "Unknown"
                                            description = rec
                                            cost = "Unknown"
                                            confidence = "Unknown"
                                        st.markdown(
                                            f'<div class="chat-recommendation">'
                                            f'<div><span class="chat-field-label">Description:</span> {description}</div>'
                                            f'<div><span class="chat-field-label">Priority:</span> <span class="priority-label">{priority}</span></div>'
                                            f'<div><span class="chat-field-label">Cost:</span> {cost}</div>'
                                            f'<div><span class="chat-field-label">Confidence:</span> <span class="confidence-level">{confidence}</span></div>'
                                            f'</div>',
                                            unsafe_allow_html=True,
                                        )
                                if "additional_considerations" in rec_data:
                                    st.markdown(
                                        f'<div style="margin-top: 8px;"><span class="chat-field-label">Additional Considerations:</span> {rec_data["additional_considerations"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if "confidence" in rec_data:
                                    st.markdown(
                                        f'<div><span class="chat-field-label">Confidence:</span> {rec_data["confidence"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                            elif "response" in response_dict:
                                response_text = response_dict["response"]
                                numbered_section = re.search(
                                    r"(\d+\.\s+\*\*.*?**:.*?(?=\n\d+\.|$|\n\n))", response_text, re.DOTALL
                                )
                                if numbered_section:
                                    considerations_start = numbered_section.start()
                                    intro_text = response_text[:considerations_start].strip()
                                else:
                                    considerations_start = len(response_text)
                                    intro_text = response_text.strip()
                                if intro_text:
                                    st.markdown(
                                        f'<div><span class="chat-field-label">Introduction:</span> {intro_text}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if numbered_section:
                                    considerations_text = re.findall(
                                        r"(\d+\.\s+\*\*.*?**:.*?(?=\n\d+\.|$|\n\n))", response_text, re.DOTALL
                                    )
                                    if considerations_text:
                                        st.markdown(
                                            f'<div style="margin-top: 8px;"><span class="chat-field-label">Considerations:</span></div>',
                                            unsafe_allow_html=True,
                                        )
                                        for consideration in considerations_text:
                                            consideration = consideration.strip()
                                            st.markdown(f"<div>{consideration}</div>", unsafe_allow_html=True)
                                recommendations_section = re.search(
                                    r"\*\*Next Steps & Recommendations\*\*:(.*?)(?=\n\n[A-Z]|$)", response_text, re.DOTALL
                                )
                                if recommendations_section:
                                    rec_text = recommendations_section.group(1).strip()
                                    recommendations = re.findall(
                                        r"\d+\.\s+\*\*.*?**.*?($$     .*?     $$:\s*(.*?)\s*$$     (.*?)     $$\s*-\s*Confidence:\s*(.*?))(?=\n\d+\.|$)",
                                        rec_text,
                                        re.DOTALL,
                                    )
                                    if recommendations:
                                        st.markdown(
                                            f'<div style="margin-top: 8px;"><span class="chat-field-label">Recommendations:</span></div>',
                                            unsafe_allow_html=True,
                                        )
                                        for rec in recommendations:
                                            full_rec, description, cost, confidence = rec
                                            priority_match = re.match(r"$$     (.*?)     $$:", full_rec)
                                            priority = priority_match.group(0) if priority_match else "Unknown"
                                            st.markdown(
                                                f'<div class="chat-recommendation">'
                                                f'<div><span class="chat-field-label">Description:</span> {description}</div>'
                                                f'<div><span class="chat-field-label">Priority:</span> <span class="priority-label">{priority}</span></div>'
                                                f'<div><span class="chat-field-label">Cost:</span> {cost}</div>'
                                                f'<div><span class="chat-field-label">Confidence:</span> <span class="confidence-level">{confidence}</span></div>'
                                                f'</div>',
                                                unsafe_allow_html=True,
                                            )
                                if recommendations_section:
                                    addl_start = recommendations_section.end()
                                    addl_text = response_text[addl_start:].strip()
                                    if addl_text and addl_text not in disclaimer:
                                        st.markdown(
                                            f'<div style="margin-top: 8px;"><span class="chat-field-label">Additional Considerations:</span> {addl_text}</div>',
                                            unsafe_allow_html=True,
                                        )
                            elif "geotechnical_assessment" in response_dict:
                                assessment = response_dict["geotechnical_assessment"]
                                if "foundation_recommendation_explanation" in assessment:
                                    st.markdown(
                                        f'<div><span class="chat-field-label">Explanation:</span> {assessment["foundation_recommendation_explanation"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if "technical_details" in assessment:
                                    tech_details = assessment["technical_details"]
                                    st.markdown(
                                        '<div style="margin-top: 8px;"><span class="chat-field-label">Technical Details:</span></div>',
                                        unsafe_allow_html=True,
                                    )
                                    for key, value in tech_details.items():
                                        if key == "recommendation_priority":
                                            match = re.match(
                                                r"$$     (.*?)     $$:\s*(.*?)\s*$$     (.*?)     $$\s*-\s*Confidence:\s*(.*?)$",
                                                value,
                                            )
                                            if match:
                                                priority = f"[{match.group(1)}]"
                                                description = match.group(2).strip()
                                                cost = match.group(3).strip()
                                                confidence = match.group(4).strip()
                                            else:
                                                priority = "Unknown"
                                                description = value
                                                cost = "Unknown"
                                                confidence = "Unknown"
                                            st.markdown(
                                                f'<div class="chat-recommendation">'
                                                f'<div><span class="chat-field-label">Description:</span> {description}</div>'
                                                f'<div><span class="chat-field-label">Priority:</span> <span class="priority-label">{priority}</span></div>'
                                                f'<div><span class="chat-field-label">Cost:</span> {cost}</div>'
                                                f'<div><span class="chat-field-label">Confidence:</span> <span class="confidence-level">{confidence}</span></div>'
                                                f'</div>',
                                                unsafe_allow_html=True,
                                            )
                                        else:
                                            st.markdown(
                                                f'<div><span class="chat-field-label">{key.replace("_", " ").title()}:</span> {value}</div>',
                                                unsafe_allow_html=True,
                                            )
                                if "additional_considerations" in assessment:
                                    st.markdown(
                                        f'<div style="margin-top: 8px;"><span class="chat-field-label">Additional Considerations:</span> {assessment["additional_considerations"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                if "confidence_level" in assessment:
                                    st.markdown(
                                        f'<div><span class="chat-field-label">Confidence Level:</span> {assessment["confidence_level"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                        else:
                            st.markdown(f"<div>{response_clean}</div>", unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)
                        if disclaimer:
                            st.markdown(f'<div class="chat-disclaimer">{disclaimer}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        logging.error(f"Error parsing chat response: {e}")
                        st.markdown(
                            f'<div class="assistant-message"><strong>Assistant:</strong> {answer}</div>',
                            unsafe_allow_html=True,
                        )

                    if st.button("Flag This", key=f"chat_{idx}_{question[:50]}"):
                        log_feedback(question, answer, "User flagged as inconsistent")
                        st.success("Feedback logged.")

                    st.markdown("---")

            def handle_question():
                user_question = st.session_state.chat_input
                if user_question and user_question not in [q for q, _ in st.session_state.chat_history]:
                    answer = chat_with_report(
                        st.session_state.feasibility_report, user_question, st.session_state.chat_history
                    )
                    st.session_state.chat_history.append((user_question, answer))
                    st.session_state.chat_input = ""

            st.text_input(
                "Ask a Question",
                placeholder="e.g., What does the seismic hazard mean?",
                key="chat_input",
                on_change=handle_question,
            )
        else:
            st.warning("No valid feasibility report available. Please complete an analysis in the 'Analysis' tab first.")

if __name__ == "__main__":
    main()