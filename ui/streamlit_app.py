import os
import uuid
import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080").strip().rstrip("/")

st.set_page_config(page_title="Saman Salamat Bot (Staging)", page_icon="🟩", layout="wide")


@st.cache_data(ttl=10)
def check_health(url: str):
    try:
        r = requests.get(f"{url}/healthz", timeout=3)
        return r.status_code, r.text
    except Exception as e:
        return None, str(e)


def call_bot(url: str, question: str, timeout: int = 120):
    payload = {"question": question}
    r = requests.post(f"{url}/answer", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def wrap_html_with_style(html: str, text_color: str, link_color: str, font_family: str, nonce: str) -> str:
    html = html or ""
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <!-- nonce:{nonce} -->
    <style>
      :root {{
        color-scheme: dark;
      }}
      body {{
        margin: 0;
        padding: 16px;
        background: transparent;
      }}
      #answer-root {{
        color: {text_color} !important;
        font-family: {font_family};
        line-height: 1.9;
        font-size: 16px;
        direction: rtl;
        text-align: right;
        white-space: normal;
        word-break: break-word;
      }}
      #answer-root a {{
        color: {link_color} !important;
        text-decoration: underline;
      }}
      /* override hard-coded inline colors if any */
      #answer-root * {{
        color: inherit !important;
      }}
    </style>
  </head>
  <body>
    <div id="answer-root">{html}</div>
  </body>
</html>
"""


st.title("🟩 Saman Salamat Bot — Staging UI (Streamlit)")
st.caption("Temporary frontend for staging tests. Calls your FastAPI /answer endpoint and renders HTML output.")

colA, colB = st.columns([2, 1], vertical_alignment="top")

with colB:
    st.subheader("Status")

    api_url = st.text_input("API_BASE_URL", value=API_BASE_URL)
    API_BASE_URL = api_url.strip().rstrip("/")

    code, text = check_health(API_BASE_URL)
    if code == 200:
        st.success("API is up ✅")
    else:
        st.warning("API not reachable ⚠️")

    with st.expander("Health response", expanded=False):
        st.code(f"status={code}\n{text}")

    st.subheader("Answer style")
    ans_color = st.color_picker("Answer text color", value="#EDEDED")
    link_color = st.color_picker("Link color", value="#8AB4F8")
    font_family = st.text_input(
        "Font family",
        value='"Vazirmatn","Vazir","IRANSans","Tahoma","Arial",sans-serif'
    )

    show_raw = st.toggle("Show raw HTML", value=False)
    show_json = st.toggle("Show full JSON response", value=False)

    if st.button("Apply style / Re-render"):
        st.session_state["render_nonce"] = str(uuid.uuid4())


with colA:
    st.subheader("Ask")
    q = st.text_area("Your question", value="", placeholder="سوال خود را تایپ کنید...", height=120)

    c1, c2, _ = st.columns([1, 1, 3])
    ask_btn = c1.button("Ask", type="primary")
    clear_btn = c2.button("Clear")

    if clear_btn:
        st.session_state.pop("last_response", None)
        st.session_state["render_nonce"] = str(uuid.uuid4())
        st.rerun()

    if ask_btn:
        if not q.strip():
            st.error("Please type a question.")
        else:
            with st.spinner("Calling API..."):
                try:
                    data = call_bot(API_BASE_URL, q.strip())
                    st.session_state["last_response"] = data
                    st.session_state["render_nonce"] = str(uuid.uuid4())
                except requests.exceptions.HTTPError as e:
                    st.error(f"HTTP error: {getattr(e.response, 'status_code', None)}")
                    st.code(getattr(e.response, "text", ""))
                except Exception as e:
                    st.error(f"Error: {e}")

    data = st.session_state.get("last_response")
    if data:
        html = data.get("Answer", "")
        status = data.get("status", "")

        st.markdown("---")
        st.subheader(f"Rendered answer (status: {status})")

        nonce = st.session_state.get("render_nonce", "init")
        wrapped = wrap_html_with_style(
            html=html,
            text_color=ans_color,
            link_color=link_color,
            font_family=font_family,
            nonce=nonce,
        )

        components.html(wrapped, height=360, scrolling=True)

        if show_raw:
            st.subheader("Raw HTML (from API)")
            st.code(html)

        if show_json:
            st.subheader("Full JSON")
            st.json(data)

