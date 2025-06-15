import streamlit as st
import json
import os
from utils.utils import sidebar_logo

sidebar_logo()

USERS_FILE = "users.json"

# Load users data or create file if missing
def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

# Save users data to file
def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def signup(users):
    st.subheader("Create a new account")
    new_username = st.text_input("Choose a username", key="signup_username")
    new_password = st.text_input("Choose a password", type="password", key="signup_password")
    signup_btn = st.button("Sign Up")

    if signup_btn:
        if not new_username or not new_password:
            st.error("Please fill both username and password")
            return False
        if new_username in users:
            st.error("Username already exists. Please choose another.")
            return False
        users[new_username] = {"password": new_password}
        save_users(users)
        st.success("Account created successfully! Please log in.")
        return True
    return False

def login(users):
    st.subheader("Log in to your account")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    login_btn = st.button("Login")

    if login_btn:
        if username in users and users[username]["password"] == password:
            st.success(f"Welcome back, {username}!")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
        else:
            st.error("Incorrect username or password")

def logout():
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""

def main():
    st.title("Login & Signup App")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""

    users = load_users()

    if st.session_state["logged_in"]:
        st.write(f"👋 You are logged in as **{st.session_state['username']}**")
        logout()
    else:
        page = st.sidebar.selectbox("Select Page", ["Login", "Signup"])
        if page == "Login":
            login(users)
        else:
            if signup(users):
                st.sidebar.success("Account created! Switch to login.")

if __name__ == "__main__":
    main()