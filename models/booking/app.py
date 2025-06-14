import streamlit as st
from datetime import datetime
from db import get_lawyers, save_booking, get_bookings

st.set_page_config(page_title="Lawyer Booking", page_icon="⚖️")
st.title("⚖️ Book a Lawyer")

# Load lawyer data
lawyers = get_lawyers()
if not lawyers:
    st.error("No lawyers available.")
    st.stop()

# Lawyer selection
lawyer_names = [f"{l['name']} ({l['specialty']})" for l in lawyers]
selected_index = st.selectbox("Choose a lawyer:", range(len(lawyers)), format_func=lambda i: lawyer_names[i])
selected_lawyer = lawyers[selected_index]

# Booking form
with st.form("booking_form"):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    date = st.date_input("Preferred Date")
    time = st.time_input("Preferred Time")

    submitted = st.form_submit_button("Book Appointment")

    if submitted:
        if not name or not email:
            st.error("Please fill all fields.")
        else:
            booking = {
                "user": name,
                "email": email,
                "lawyer": selected_lawyer["name"],
                "specialty": selected_lawyer["specialty"],
                "date": date.strftime("%Y-%m-%d"),
                "time": time.strftime("%H:%M"),
                "timestamp": datetime.now().isoformat()
            }
            save_booking(booking)
            st.success(" Booking Confirmed!")
            st.json(booking)

# View booking history (optional)
with st.expander("📋 View Your Bookings"):
    bookings = get_bookings()
    for b in bookings[::-1]:  # show latest first
        st.write(f"**{b['user']}** booked **{b['lawyer']}** on `{b['date']} at {b['time']}`")
