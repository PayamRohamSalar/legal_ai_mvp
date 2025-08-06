# streamlit_app/app.py
import streamlit as st
import requests
import os

st.title("⚖️ دستیار هوشمند حقوقی")
st.caption("مبتنی بر قوانین و مقررات حوزه پژوهش و فناوری ایران")

# Get backend URL from environment variable, with a default for local testing
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_ENDPOINT = f"{BACKEND_URL}/api/v1/health"

st.sidebar.header("وضعیت سرویس")
try:
    response = requests.get(API_ENDPOINT, timeout=5)
    if response.status_code == 200:
        status = response.json()
        st.sidebar.success(f"سرویس Backend فعال است.\n- نام برنامه: {status.get('app_name', 'N/A')}\n- نسخه: {status.get('version', 'N/A')}")
    else:
        st.sidebar.error(f"سرویس Backend در دسترس نیست. کد وضعیت: {response.status_code}")
except requests.exceptions.RequestException:
    st.sidebar.error("اتصال به سرویس Backend برقرار نشد.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("سوال خود را در مورد قوانین پژوهش و فناوری بپرسید..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("در حال فکر کردن...")
        # In the next phase, we will call the actual chat API here.
        # For now, we just show a placeholder response.
        full_response = "پاسخ نمونه: در فاز بعدی، پاسخ واقعی از مدل زبان بزرگ در اینجا نمایش داده خواهد شد."
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})