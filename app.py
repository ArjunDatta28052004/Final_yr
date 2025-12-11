import streamlit as st
import pandas as pd
import requests
import os
import time

# -------------------------------------------------------------
# CONFIG - BACKEND URL (ngrok or local)
# -------------------------------------------------------------
# Prefer secrets (e.g., on Streamlit Cloud set: BACKEND_URL = "https://xxx.ngrok-free.app")
BACKEND_URL = "https://candent-shasta-casuistically.ngrok-free.dev"

# -------------------------------------------------------------
# Streamlit Config
# -------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Universal AI RAG Insurance Validator"
)

# -------------------------------------------------------------
# Persistent State
# -------------------------------------------------------------
if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "policy_name" not in st.session_state:
    st.session_state.policy_name = ""

# -------------------------------------------------------------
# UI Header
# -------------------------------------------------------------
st.title("🛡️ Universal AI Insurance Claim Validator")
st.subheader("Powered by Ollama + Hugging Face — Works with *ANY* Policy")
st.caption(f"Backend: {BACKEND_URL}")

# -------------------------------------------------------------
# Sidebar – Upload Policy
# -------------------------------------------------------------
st.sidebar.markdown("### 📄 Step 1: Upload Policy Document")
st.sidebar.caption("The uploaded PDF will be sent to the backend for ingestion (Ollama must be running on the machine behind the backend).")

uploaded_file = st.sidebar.file_uploader(
    "Upload Insurance Policy (PDF):",
    type=["pdf"],
    disabled=st.session_state.ingested,
    help="Upload any insurance policy: health, auto, home, life, etc."
)

# -------------------------------------------------------------
# Step 1 — Ingest Policy (Build KB) via file upload to backend
# -------------------------------------------------------------
if uploaded_file and not st.session_state.ingested:
    st.session_state.policy_name = uploaded_file.name

    if st.sidebar.button("🔨 Build Knowledge Base", type="primary"):
        try:
            with st.spinner("🔄 Uploading & Processing Policy…"):
                # Send file to backend as multipart
                files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), "application/pdf")}
                resp = requests.post(f"{BACKEND_URL}/ingest_file", files=files, timeout=300)

                if resp.status_code != 200:
                    st.sidebar.error(f"❌ Ingest failed: {resp.status_code} {resp.text}")
                else:
                    data = resp.json()
                    if data.get("error"):
                        st.sidebar.error(f"❌ Error: {data.get('error')}")
                    else:
                        st.session_state.ingested = True
                        st.sidebar.success("✅ Knowledge Base Built!")
                        st.sidebar.caption(f"📊 {data.get('pages')} pages → {data.get('chunks')} chunks")
                        summary = data.get("summary")
                        if summary:
                            st.sidebar.markdown("**📋 Policy Summary:**")
                            st.sidebar.info(summary)

        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            st.sidebar.caption("Check: 1) PDF validity 2) Backend reachable (ngrok running)")

elif st.session_state.ingested:
    st.sidebar.success(f"Policy Loaded: {st.session_state.policy_name}")
    # Optionally fetch stored summary from backend if needed

# -------------------------------------------------------------
# Step 2 — Claim Validation Interface
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 Step 2: Validate Claim")

if st.session_state.ingested:

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Enter claim details below:**")

    with col2:
        example_type = st.selectbox(
            "Load example:",
            ["Custom", "Health Insurance", "Auto Insurance", "Home Insurance", "Life Insurance"]
        )

    examples = {
        "Health Insurance": "45-year-old patient in Mumbai underwent appendectomy surgery on Oct 15, 2024. Required 2 days hospitalization. Total bill ₹75,000.",
        "Auto Insurance": "Collision in Bangalore on Nov 20, 2024. Vehicle damage ₹1,50,000. Comprehensive policy active 2 years.",
        "Home Insurance": "Fire damaged kitchen on Dec 1, 2024 in Delhi. Damage estimate ₹3,00,000.",
        "Life Insurance": "60-year-old suffered heart attack on Nov 25, 2024. Claiming critical illness benefit ₹10,00,000.",
        "Custom": "55-year-old male hospitalized 48 hours for viral fever. Policy active 5 years."
    }

    default_text = examples.get(example_type, examples["Custom"])

    claim_text = st.text_area(
        "Claim Description:",
        default_text,
        height=140
    )

    col_a, col_b, _ = st.columns([1, 1, 4])

    with col_a:
        validate_button = st.button("✅ Validate Claim", type="primary")

    with col_b:
        if st.button("🔄 Clear"):
            st.rerun()

    # -------------------------------------------------------------
    # Process Claim
    # -------------------------------------------------------------
    if validate_button and claim_text.strip():
        with st.spinner("🤖 Validating against policy…"):
            try:
                start = time.time()
                resp = requests.post(f"{BACKEND_URL}/validate", json={"claim_text": claim_text}, timeout=120)
                if resp.status_code != 200:
                    st.error(f"❌ Validation failed: {resp.status_code} {resp.text}")
                else:
                    result = resp.json()
                    end = time.time()

                    st.markdown("---")
                    st.markdown("## 📊 Validation Result")

                    is_valid = result.get("is_valid", False)

                    col_r, col_t = st.columns([4, 1])
                    with col_r:
                        if is_valid:
                            st.success("### ✅ CLAIM IS VALID")
                        else:
                            st.error("### ❌ CLAIM IS NOT VALID")

                    with col_t:
                        st.metric("Time", f"{end - start:.2f}s")

                    # Explanation
                    st.markdown("**Decision Explanation:**")
                    st.info(result.get("decision_reason", "No reason provided."))

                    # -------------------------------------------------------------
                    # Tabs
                    # -------------------------------------------------------------
                    t1, t2, t3, t4 = st.tabs([
                        "📋 Extracted Details",
                        "✓ Validation Checks",
                        "📜 Policy Context",
                        "🔧 Raw Output"
                    ])

                    # Tab 1 — Entities
                    with t1:
                        entities = result.get("extracted_entities", {})
                        if entities:
                            df = pd.DataFrame(
                                [(k, v) for k, v in entities.items()],
                                columns=["Field", "Value"]
                            )
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No entities extracted.")

                    # Tab 2 — Checks
                    with t2:
                        checks = result.get("validation_checks", [])
                        if not checks:
                            st.info("No checks available.")
                        else:
                            for c in checks:
                                status = c.get("status", "").upper()
                                name = c.get("check", "")
                                ref = c.get("policy_reference", "N/A")

                                if status == "PASS":
                                    st.success(f"✓ **{name}**\n\n📖 {ref}")
                                elif status == "FAIL":
                                    st.error(f"✗ **{name}**\n\n📖 {ref}")
                                else:
                                    st.info(f"? **{name}** - {status}\n\n📖 {ref}")

                    # Tab 3 — Policy Context
                    with t3:
                        clauses = result.get("policy_clauses_used", [])
                        if not clauses:
                            st.warning("No clauses used.")
                        else:
                            for i, c in enumerate(clauses):
                                with st.expander(f"📄 Clause {i+1}", expanded=(i == 0)):
                                    st.text(c)

                    # Tab 4 — Raw JSON
                    with t4:
                        st.json(result)

            except Exception as e:
                st.error(f"❌ Validation Error: {str(e)}")
                with st.expander("🔍 Details"):
                    st.exception(e)

    elif validate_button:
        st.warning("⚠ Please enter claim details before validating.")

else:
    st.info("👆 Upload a policy document to begin.")
    st.markdown("---")

    st.markdown("### ✨ How It Works")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**1️⃣ Upload Policy**")
        st.caption("System reads and builds searchable knowledge base.")

    with c2:
        st.markdown("**2️⃣ Enter Claim**")
        st.caption("Describe hospitalization, bills, vehicle damage, etc.")

    with c3:
        st.markdown("**3️⃣ Get Validation**")
        st.caption("AI checks claim against policy rules.")

# -------------------------------------------------------------
# Reset
# -------------------------------------------------------------
if st.session_state.ingested:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Application"):
        st.session_state.ingested = False
        st.session_state.policy_name = ""
        try:
            if os.path.exists("temp_policy.pdf"):
                os.remove("temp_policy.pdf")
        except:
            pass
        st.rerun()

st.markdown("---")
st.caption("🤖 Powered by Phi-3 + HuggingFace + LangChain + Streamlit")
