import streamlit as st
from PIL import Image
from utils.utils import sidebar_logo

st.set_page_config(page_title="Satyanistha", layout="wide")

sidebar_logo()

# Page config

# Custom CSS Styling
st.markdown("""
    <style>
        html, body {
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
        }

        /* Navigation Bar */
        .nav-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: white;
            padding: 20px 60px;
            border-bottom: 1px solid #ddd;
        }

        .nav-links a {
            margin: 0 20px;
            text-decoration: none;
            color: #204939;
            font-weight: 600;
        }

        .nav-links a:hover {
            color: #0d2e22;
        }

        /* Hero Section */
        .hero {
            text-align: center;
            padding: 80px 20px 50px;
            background-color: #f5f5f5;
        }

        .hero h1 {
            font-size: 56px;
            font-weight: 800;
            color: #204939;
            margin-bottom: 10px;
        }

        .hero h3 {
            font-size: 22px;
            color: #333;
        }

        .hero button {
            padding: 12px 30px;
            margin-top: 30px;
            background-color: #204939;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .hero button:hover {
            background-color: #18352a;
        }

        /* Image Section */
        .styled-img {
            display: flex;
            justify-content: center;
            gap: 40px;
            flex-wrap: wrap;
            padding: 40px 20px;
            background-color: #fff;
        }

        .styled-img img {
            width: 300px;
            height: 200px;
            object-fit: cover;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .styled-img img:hover {
            transform: scale(1.03);
        }

        /* Feature Cards */
        .features {
            display: flex;
            justify-content: center;
            gap: 30px;
            padding: 60px 20px;
            flex-wrap: wrap;
            background-color: #f9f9f9;
        }

        .feature-box {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 30px 20px;
            width: 220px;
            text-align: center;
            box-shadow: 0px 6px 16px rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
            border: 1px solid #e2e2e2;
        }

        .feature-box:hover {
            transform: translateY(-6px);
            border-color: #204939;
        }

        .feature-box i {
            font-size: 28px;
            color: #204939;
        }

        .feature-box h4 {
            margin: 15px 0 10px;
            font-size: 18px;
            color: #204939;
        }

        .feature-box p {
            font-size: 14px;
            color: #555;
        }

        /* Footer */
        .footer {
            background-color: #204939;
            color: white;
            padding: 40px 50px;
            text-align: center;
        }

        .footer p {
            margin: 5px 0;
        }

        .footer a {
            color: #d0d0d0;
            text-decoration: none;
            margin: 0 12px;
        }

        .footer a:hover {
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation Bar
st.markdown("""
    <div class="nav-bar">
        <div class="nav-links">
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Why Us?</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero">
        <h1>Satyanistha</h1>
        <h3>Hajur ko Kanuni Sallahkar</h3>
        <form action="" method="get">
            <button type="submit" name="chat_button">Chat With Us</button>
        </form>
    </div>
""", unsafe_allow_html=True)

# Detecting if the button was clicked using query params (simple trick)
if st.query_params.get("chat_button") is not None:
    st.switch_page("main.py")  # or the correct path like "pages/Chat.py"

# Image Section (fixed for local images)
st.markdown("<div class='styled-img'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.image(Image.open("../models/logo.png"), width=300)
with col2:
    # st.image(Image.open("/Users/aakritidahal/Desktop/2-Bit copy/models/booking/image/logo.jpeg"), width=300)
    st.image(Image.open("../models/logo.png"), width=300)
with col3:
    # st.image(Image.open("/Users/aakritidahal/Desktop/2-Bit copy/models/booking/image/logo.jpeg"), width=300)
    st.image(Image.open("../models/logo.png"), width=300)

st.markdown("</div>", unsafe_allow_html=True)

# Features Section
st.markdown("""
    <div class="features">
        <div class="feature-box">
            <i>⚖️</i>
            <h4>Expert Advice</h4>
            <p>Get reliable legal consultation from verified experts.</p>
        </div>
        <div class="feature-box">
            <i>🗣️</i>
            <h4>Local Language</h4>
            <p>We explain your rights in a language you understand.</p>
        </div>
        <div class="feature-box">
            <i>🔒</i>
            <h4>Confidential</h4>
            <p>Your legal queries are handled with privacy and care.</p>
        </div>
        <div class="feature-box">
            <i>💰</i>
            <h4>Reliable</h4>
            <p>Legal help whenever you need.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Footer Section
st.markdown("""
    <div class="footer">
        <p>&copy; 2025 Satyanistha. All rights reserved.</p>
        <p>
            <a href="#">Privacy Policy</a> |
            <a href="#">Terms of Service</a> |
            <a href="#">Contact Us</a>
        </p>
    </div>
""", unsafe_allow_html=True)