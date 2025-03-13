import streamlit as st
from models import Address, Coordinates, Property, SlopeData, EnvironmentalCheck, FeasibilityReport
from geo_processing import geocode_address, extract_property, calculate_slope, check_environmental_hazards, create_map
from gemini_analysis import analyze_location, analyze_slope, generate_feasibility_report, chat_with_report

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

def perform_analysis(street: str, zip_code: str) -> None:
    """Perform the full property analysis and store results in session state."""
    # Step 1: Geocode address
    address = Address(street=street, zip_code=zip_code if zip_code else None)
    coordinates = geocode_address(address)
    if not coordinates:
        st.error("Failed to geocode address. Please check the input and try again.")
        return
    st.session_state.coordinates = coordinates
    st.success(f"Geocoded Address: ({coordinates.latitude}, {coordinates.longitude})")

    # Step 2: Extract property data
    property = extract_property(coordinates)
    if not property:
        st.error("No property found at the given coordinates.")
        return
    st.session_state.property = property

    # Step 3: Calculate slope data
    slope_data = calculate_slope(property)
    if not slope_data:
        st.error("Failed to calculate slope data.")
        return
    st.session_state.slope_data = slope_data

    # Step 4: Check environmental hazards
    environmental_check = check_environmental_hazards(property)
    if not environmental_check:
        st.error("Failed to check environmental hazards.")
        return
    st.session_state.environmental_check = environmental_check

    # Step 5: Gemini location analysis
    location_analysis = analyze_location(environmental_check)

    # Step 6: Gemini slope analysis
    slope_analysis = analyze_slope(slope_data)

    # Step 7: Gemini feasibility report
    feasibility_report = generate_feasibility_report(location_analysis, slope_analysis)
    st.session_state.feasibility_report = feasibility_report

def display_report():
    """Display the feasibility report and map from session state if all required data is present."""
    required_keys = ["coordinates", "property", "feasibility_report"]
    if all(key in st.session_state and st.session_state[key] is not None for key in required_keys):
        report = st.session_state.feasibility_report
        # Additional check to ensure report has necessary attributes
        if hasattr(report, "location_analysis") and hasattr(report, "slope_analysis"):
            st.subheader("Property Map")
            create_map(st.session_state.coordinates, st.session_state.property, GEOJSON_FILES)

            st.subheader("Feasibility Report")
            st.write("**Location Analysis Summary:**", report.location_analysis.summary)
            st.write("**Location Recommendations:**")
            for rec in report.location_analysis.recommendations:
                st.write(f"- {rec}")
            st.write("**Slope Analysis Summary:**", report.slope_analysis.summary)
            st.write("**Slope Recommendations:**")
            for rec in report.slope_analysis.recommendations:
                st.write(f"- {rec}")
            st.write("**Overall Feasibility:**", report.overall_feasibility)
            st.write("**Detailed Recommendations:**")
            for rec in report.detailed_recommendations:
                st.write(f"- {rec}")
        else:
            st.warning("Feasibility report is incomplete. Please re-run the analysis.")
    else:
        st.info("No analysis results available yet. Please enter an address and click 'Analyze Property'.")

def main():
    st.title("Geotechnical Engineering Assistant - Mercer Island, WA")

    # Initialize session state for analysis results and chat history
    if "coordinates" not in st.session_state:
        st.session_state.coordinates = None
    if "property" not in st.session_state:
        st.session_state.property = None
    if "slope_data" not in st.session_state:
        st.session_state.slope_data = None
    if "environmental_check" not in st.session_state:
        st.session_state.environmental_check = None
    if "feasibility_report" not in st.session_state:
        st.session_state.feasibility_report = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List to store (question, answer) tuples
    if "street_input" not in st.session_state:
        st.session_state.street_input = ""
    if "zip_input" not in st.session_state:
        st.session_state.zip_input = ""

    # Sidebar for address input
    with st.sidebar:
        st.header("Address Input")
        street = st.text_input("Street Address", value=st.session_state.street_input, placeholder="e.g., 1234 Main St", key="street")
        zip_code = st.text_input("ZIP Code (optional)", value=st.session_state.zip_input, placeholder="e.g., 98040", key="zip")
        analyze_button = st.button("Analyze Property")

        # Update session state with input values
        st.session_state.street_input = street
        st.session_state.zip_input = zip_code

        if analyze_button and street:
            perform_analysis(street, zip_code)

    # Create tabs for Analysis and Chat
    analysis_tab, chat_tab = st.tabs(["Analysis", "Chat"])

    # Analysis Tab
    with analysis_tab:
        display_report()

    # Chat Tab
    with chat_tab:
        if st.session_state.feasibility_report and hasattr(st.session_state.feasibility_report, "location_analysis"):
            st.subheader("Chat with Your Feasibility Report")

            # Create a container for the chat history with a fixed height and scroll
            chat_container = st.container()
            with chat_container:
                # Display chat history in a conversational format
                for question, answer in st.session_state.chat_history:
                    st.markdown(f"**You:** {question}")
                    st.markdown(f"**Assistant:** {answer}")
                    st.markdown("---")

            # Input for new question with a callback to handle submission
            def handle_question():
                user_question = st.session_state.chat_input
                if user_question and user_question not in [q for q, _ in st.session_state.chat_history]:
                    # Pass the chat history to chat_with_report
                    answer = chat_with_report(st.session_state.feasibility_report, user_question, st.session_state.chat_history)
                    st.session_state.chat_history.append((user_question, answer))
                    # Clear the input
                    st.session_state.chat_input = ""

            st.text_input("Ask a Question", placeholder="e.g., What does the seismic hazard mean?", key="chat_input", on_change=handle_question)
        else:
            st.warning("No valid feasibility report available. Please complete an analysis in the 'Analysis' tab first.")

if __name__ == "__main__":
    main()