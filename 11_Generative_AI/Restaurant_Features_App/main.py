import streamlit as st
import langchain_helper as lh

# Page configuration
st.set_page_config(
    page_title="Restaurant Name Generator", page_icon="🍽️", layout="centered"
)

# Title and description
st.title("Restaurant Name Generator")
st.markdown(
    "Generate a fancy restaurant name, menu items, ambiance, and signature dish for your chosen cuisine. Customize with dietary preferences!"
)

# Sidebar for inputs
with st.sidebar:
    st.header("Customize Your Restaurant")
    cuisine = st.selectbox(
        "Pick a Cuisine",
        ["Indian", "Italian", "Mexican", "Arabic", "American", "Other"],
    )
    if cuisine == "Other":
        cuisine = st.text_input("Enter Custom Cuisine")
    preferences = st.text_area(
        "Dietary Preferences (e.g., vegan, gluten-free)",
        placeholder="No specific preferences",
    )

# Generate button
if st.button("Generate Restaurant"):
    if cuisine:
        with st.spinner("Generating restaurant details..."):
            try:
                response = lh.generate_restaurant_features(cuisine, preferences)
                if response["restaurant_name"] == "Error":
                    st.error(response["menu_items"])
                else:
                    st.header(response["restaurant_name"].strip())
                    st.subheader("Menu Items")
                    menu_items = response["menu_items"].strip().split(",")
                    for item in menu_items:
                        st.write(f"- {item.strip()}")
                    st.subheader("Ambiance")
                    st.write(response["ambiance"].strip())
                    st.subheader("Signature Dish")
                    st.write(response["signature_dish"].strip())
                    # Display conversation history
                    st.subheader("Conversation History")
                    st.write(lh.memory.load_memory_variables({})["chat_history"])
            except Exception as e:
                st.error(f"Error generating restaurant: {str(e)}")
    else:
        st.warning("Please select or enter a cuisine.")
