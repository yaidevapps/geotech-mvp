import streamlit as st
from models import Address, Coordinates, Property, SlopeData, EnvironmentalCheck, FeasibilityReport
from geo_processing import geocode_address, extract_property, calculate_slope, check_environmental_hazards, create_map
from gemini_analysis import analyze_location, analyze_slope, generate_feasibility_report, chat_with_report
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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

    property = extract_property(coordinates)
    if not property:
        st.error("No property found at the given coordinates.")
        return
    st.session_state.property = property

    slope_data = calculate_slope(property)
    if not slope_data:
        st.error("Failed to calculate slope data.")
        return
    logging.debug(f"Slope Data - Average: {slope_data.average_slope}, Max: {slope_data.max_slope}")
    st.session_state.slope_data = slope_data

    environmental_check = check_environmental_hazards(property)
    if not environmental_check:
        st.error("Failed to check environmental hazards.")
        return
    st.session_state.environmental_check = environmental_check

    with st.spinner("Performing location analysis..."):
        location_analysis = analyze_location(
            latitude=coordinates.latitude,
            longitude=coordinates.longitude,
            address=address_str
        )
    st.session_state.location_analysis = location_analysis

    with st.spinner("Performing slope analysis..."):
        avg_slope_deg = slope_data.average_slope if hasattr(slope_data, 'average_slope') else 0
        avg_slope_percent = np.tan(np.radians(avg_slope_deg)) * 100
        dist = 10.0  # Default distance in meters
        elev_diff_m = (avg_slope_percent / 100) * dist
        elev_diff_ft = elev_diff_m * 3.28084
        slope_analysis = analyze_slope(
            slope=avg_slope_percent,
            elevation_diff=elev_diff_ft,
            distance=dist
        )
    st.session_state.slope_analysis = slope_analysis

    with st.spinner("Generating feasibility report..."):
        environmental_hazards = {
            "erosion": environmental_check.erosion,
            "potential_slide": environmental_check.potential_slide,
            "seismic": environmental_check.seismic,
            "steep_slope": environmental_check.steep_slope,
            "watercourse": environmental_check.watercourse
        }
        feasibility_report = generate_feasibility_report(
            address=address_str,
            slope_analysis=slope_analysis,
            location_analysis=location_analysis,
            environmental_hazards=environmental_hazards
        )
    st.session_state.feasibility_report = feasibility_report

def display_report():
    """Display the feasibility report with expandable sections and feedback options."""
    required_keys = ["coordinates", "property", "feasibility_report"]
    if not all(key in st.session_state and st.session_state[key] is not None for key in required_keys):
        st.info("No analysis results available yet. Please enter an address and click 'Analyze Property'.")
        return

    report = st.session_state.feasibility_report
    if not (hasattr(report, "location_analysis") and hasattr(report, "slope_analysis")):
        st.warning("Feasibility report is incomplete. Please re-run the analysis.")
        return

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
                    unsafe_allow_html=True
                )
                create_map(st.session_state.coordinates, st.session_state.property, GEOJSON_FILES)

    st.markdown(f"**Overall Feasibility:** {report.overall_feasibility}", unsafe_allow_html=True)

    with st.expander("Hazard Layer Information", expanded=True):
        if hasattr(report, 'hazard_layers'):
            for hazard in report.hazard_layers:
                icon = "🟢" if "Not Present" in hazard else "🔴"
                st.write(f"{icon} {hazard}")
        else:
            st.write("Hazard layer information unavailable.")

    with st.expander("Location Analysis"):
        st.write("**Summary:**", report.location_analysis.summary if report.location_analysis else "Analysis unavailable")
        if report.location_analysis and hasattr(report.location_analysis, 'recommendations'):
            st.write("**Recommendations:**")
            for rec in report.location_analysis.recommendations:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"- {rec}")
                with col2:
                    if st.button("Flag This", key=f"loc_{rec[:50]}"):
                        log_feedback("Location Analysis", rec, "User flagged as inconsistent")
                        st.success("Feedback logged.")

    with st.expander("Slope Analysis"):
        st.write("**Summary:**", report.slope_analysis.summary if report.slope_analysis else "Analysis unavailable")
        if report.slope_analysis and hasattr(report.slope_analysis, 'recommendations'):
            st.write("**Recommendations:**")
            for rec in report.slope_analysis.recommendations:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"- {rec}")
                with col2:
                    if st.button("Flag This", key=f"slope_{rec[:50]}"):
                        log_feedback("Slope Analysis", rec, "User flagged as inconsistent")
                        st.success("Feedback logged.")

    with st.expander("Detailed Recommendations"):
        if hasattr(report, 'detailed_recommendations'):
            for rec in report.detailed_recommendations:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"- {rec}")
                with col2:
                    if st.button("Flag This", key=f"detail_{rec[:50]}"):
                        log_feedback("Detailed Recommendations", rec, "User flagged as inconsistent")
                        st.success("Feedback logged.")
        else:
            st.write("No detailed recommendations available.")

def main():
    st.title("Geotechnical Engineering Assistant - Mercer Island, WA")

    if "coordinates" not in st.session_state:
        st.session_state.coordinates = None
    if "address" not in st.session_state:
        st.session_state.address = ""
    if "property" not in st.session_state:
        st.session_state.property = None
    if "slope_data" not in st.session_state:
        st.session_state.slope_data = None
    if "environmental_check" not in st.session_state:
        st.session_state.environmental_check = None
    if "location_analysis" not in st.session_state:
        st.session_state.location_analysis = None
    if "slope_analysis" not in st.session_state:
        st.session_state.slope_analysis = None
    if "feasibility_report" not in st.session_state:
        st.session_state.feasibility_report = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "street_input" not in st.session_state:
        st.session_state.street_input = ""
    if "zip_input" not in st.session_state:
        st.session_state.zip_input = ""

    with st.sidebar:
        st.header("Address Input")
        street = st.text_input("Street Address", value=st.session_state.street_input, placeholder="e.g., 1234 Main St", key="street")
        zip_code = st.text_input("ZIP Code (optional)", value=st.session_state.zip_input, placeholder="e.g., 98040", key="zip")
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
            st.subheader("Chat with Your Feasibility Report")

            chat_container = st.container()
            with chat_container:
                for question, answer in st.session_state.chat_history:
                    st.markdown(f"**You:** {question}")
                    st.markdown(f"**Assistant:** {answer}")
                    st.markdown("---")

            def handle_question():
                user_question = st.session_state.chat_input
                if user_question and user_question not in [q for q, _ in st.session_state.chat_history]:
                    answer = chat_with_report(st.session_state.feasibility_report, user_question, st.session_state.chat_history)
                    st.session_state.chat_history.append((user_question, answer))
                    st.session_state.chat_input = ""

            st.text_input("Ask a Question", placeholder="e.g., What does the seismic hazard mean?", key="chat_input", on_change=handle_question)
        else:
            st.warning("No valid feasibility report available. Please complete an analysis in the 'Analysis' tab first.")

if __name__ == "__main__":
    main()