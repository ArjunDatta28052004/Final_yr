import streamlit as st
import pandas as pd
import requests
import os
import time

# CONFIG - BACKEND URL
# For ngrok development
BACKEND_URL = "https://candent-shasta-casuistically.ngrok-free.dev"

# Streamlit Config
st.set_page_config(
    layout="wide",
    page_title="AI Insurance Claim Validator",
    page_icon="🛡️"
)

# Custom CSS for better UI-
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "policy_name" not in st.session_state:
    st.session_state.policy_name = ""

if "policy_summary" not in st.session_state:
    st.session_state.policy_summary = ""

if "validation_history" not in st.session_state:
    st.session_state.validation_history = []

# UI Header
st.markdown('<div class="main-header">🛡️ AI Insurance Claim Validator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Dynamic RAG | Works with ANY Insurance Policy</div>', unsafe_allow_html=True)

# Show backend connection status
try:
    response = requests.get(f"{BACKEND_URL}/", timeout=2)
    if response.status_code == 200:
        st.success(f"✅ Connected to backend")
    else:
        st.warning(f"⚠️ Backend responding but with unexpected status: {BACKEND_URL}")
except:
    st.error(f"❌ Cannot connect to backend: {BACKEND_URL}")
    st.info("💡 Make sure the backend is running: `uvicorn api_server:app --reload`")

# Sidebar – Policy Upload
st.sidebar.markdown("## 📄 Step 1: Upload Policy")
st.sidebar.markdown("Upload any insurance policy PDF (health, auto, home, life, etc.)")

uploaded_file = st.sidebar.file_uploader(
    "Choose Policy PDF:",
    type=["pdf"],
    disabled=st.session_state.ingested,
    help="The system will analyze the policy and determine validation requirements automatically"
)

# Policy Ingestion
if uploaded_file and not st.session_state.ingested:
    st.session_state.policy_name = uploaded_file.name

    if st.sidebar.button("🔨 Analyze Policy & Build Knowledge Base", type="primary"):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            status_text.text("📤 Uploading policy...")
            progress_bar.progress(20)
            
            # Send file to backend
            files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), "application/pdf")}
            
            status_text.text("🔄 Processing document...")
            progress_bar.progress(40)
            
            resp = requests.post(
                f"{BACKEND_URL}/ingest_file", 
                files=files, 
                timeout=300
            )

            progress_bar.progress(80)
            
            if resp.status_code != 200:
                st.sidebar.error(f"❌ Ingestion failed: {resp.status_code}")
                st.sidebar.text(resp.text)
            else:
                data = resp.json()
                if data.get("error"):
                    st.sidebar.error(f"❌ Error: {data.get('error')}")
                else:
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    
                    st.session_state.ingested = True
                    st.session_state.policy_summary = data.get("summary", "")
                    
                    st.sidebar.success("✅ Knowledge Base Built Successfully!")
                    st.sidebar.metric("Pages Processed", data.get('pages', 0))
                    st.sidebar.metric("Chunks Created", data.get('chunks', 0))
                    
                    if st.session_state.policy_summary:
                        st.sidebar.markdown("### 📋 Policy Summary")
                        st.sidebar.info(st.session_state.policy_summary)
                    
                    # Clear progress indicators
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()

        except requests.exceptions.Timeout:
            st.sidebar.error("❌ Request timed out. Policy might be too large or backend is slow.")
        except requests.exceptions.ConnectionError:
            st.sidebar.error("❌ Cannot connect to backend. Make sure it's running!")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            st.sidebar.caption("Check: 1) PDF validity 2) Backend reachable 3) Ollama running")

elif st.session_state.ingested:
    st.sidebar.success(f"✅ Policy Loaded: {st.session_state.policy_name}")
    
    with st.sidebar.expander("📋 View Policy Summary"):
        st.info(st.session_state.policy_summary)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Policy Actions")
    if st.sidebar.button("🔄 Upload Different Policy"):
        st.session_state.ingested = False
        st.session_state.policy_name = ""
        st.session_state.policy_summary = ""
        st.session_state.validation_history = []
        st.rerun()

# Main Content - Claim Validation
st.markdown("---")

if st.session_state.ingested:
    st.markdown("## 🔍 Step 2: Validate Insurance Claim")
    st.markdown("Enter claim details below. The system will automatically determine what checks are needed based on the policy.")
    
    # Example selector
    col1, col2 = st.columns([4, 1])
    
    with col2:
        example_type = st.selectbox(
            "Load Example:",
            ["Custom", "Health - Surgery", "Health - Pre-existing", "Auto - Accident", 
             "Home - Fire", "Life - Critical Illness"],
            help="Select a pre-filled example or enter custom claim"
        )
    
    # Example claims
    examples = {
        "Health - Surgery": "45-year-old patient underwent appendectomy surgery on Oct 15, 2024. Hospitalized for 2 days at City Hospital. Total medical bill: ₹75,000. Policy has been active for 3 years.",
        
        "Health - Pre-existing": "58-year-old patient with diabetes (diagnosed 5 years ago) hospitalized for 4 days due to cardiac surgery. Policy active for 30 months. Total claim: ₹3,80,000. Room charges: ₹8,000 per day.",
        
        "Auto - Accident": "Collision accident in Bangalore on Nov 20, 2024. 7-year-old sedan sustained front bumper and engine damage. Estimated repair cost: ₹1,50,000. Comprehensive insurance policy active for 2 years. No-claim bonus applicable.",
        
        "Home - Fire": "Kitchen fire in residential property in Delhi on Dec 1, 2024. Fire caused by electrical short circuit. Damage to kitchen appliances and structure. Estimated damage: ₹3,00,000. Property insured for ₹50,00,000.",
        
        "Life - Critical Illness": "60-year-old male suffered heart attack on Nov 25, 2024. Hospitalized for 10 days in ICU. Claiming critical illness benefit of ₹10,00,000. Policy active for 8 years. Regular premium payments up to date.",
        
        "Custom": ""
    }
    
    default_text = examples.get(example_type, "")
    
    with col1:
        st.markdown("**Enter detailed claim information:**")
    
    claim_text = st.text_area(
        "Claim Description:",
        value=default_text,
        height=150,
        placeholder="Example: 45-year-old patient hospitalized for 3 days for emergency surgery. Policy active 2 years. Total bill: ₹1,50,000...",
        help="Include: age, procedure/incident, duration, amounts, policy details, etc."
    )

    # Action buttons
    col_a, col_b, col_c, _ = st.columns([1, 1, 1, 3])

    with col_a:
        validate_button = st.button("✅ Validate Claim", type="primary", use_container_width=True)

    with col_b:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    with col_c:
        if st.session_state.validation_history:
            show_history = st.button("📜 History", use_container_width=True)

    if clear_button:
        st.rerun()

    # Show validation history
    if st.session_state.validation_history and 'show_history' in locals() and show_history:
        with st.expander("📜 Validation History", expanded=True):
            for i, record in enumerate(reversed(st.session_state.validation_history[-5:]), 1):
                status_icon = "✅" if record["valid"] else "❌"
                st.markdown(f"**{status_icon} Validation {len(st.session_state.validation_history) - i + 1}**")
                st.caption(f"Claim: {record['claim'][:100]}...")
                st.caption(f"Result: {record['result']}")
                st.markdown("---")

    # Process Claim Validation
    if validate_button and claim_text.strip():
        
        # Add to history
        validation_record = {
            "claim": claim_text,
            "valid": None,
            "result": None
        }
        
        with st.spinner("🤖 Analyzing claim against policy..."):
            try:
                start_time = time.time()
                
                # Show progress
                progress_placeholder = st.empty()
                progress_placeholder.info("📊 Extracting claim details...")
                
                # Make validation request with increased timeout
                resp = requests.post(
                    f"{BACKEND_URL}/validate", 
                    json={"claim_text": claim_text}, 
                    timeout=300  # Increased to 5 minutes
                )
                
                progress_placeholder.info("🔍 Performing validation checks...")
                
                if resp.status_code != 200:
                    st.error(f"❌ Validation request failed: {resp.status_code}")
                    st.text(resp.text)
                else:
                    result = resp.json()
                    end_time = time.time()
                    duration = end_time - start_time

                    # create logs directory
                    log_dir = "performance_logs"
                    os.makedirs(log_dir, exist_ok=True)

                    # Save the result to a unique JSON file
                    import json
                    log_filename = f"{log_dir}/run_{int(time.time())}.json"
                    with open(log_filename, "w") as f:
                        json.dump(result, f)
                    progress_placeholder.empty()
                    
                    # Update history
                    validation_record["valid"] = result.get("is_valid", False)
                    validation_record["result"] = "Valid" if result.get("is_valid") else "Invalid"
                    st.session_state.validation_history.append(validation_record)

                    st.markdown("---")
                    st.markdown("## 📊 Validation Results")

                    is_valid = result.get("is_valid", False)

                    # Result header
                    col_result, col_time = st.columns([5, 1])
                    
                    with col_result:
                        if is_valid:
                            st.markdown('<div class="success-box"><h2 style="color: #28a745; margin: 0;">✅ CLAIM IS VALID</h2></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-box"><h2 style="color: #dc3545; margin: 0;">❌ CLAIM IS INVALID</h2></div>', unsafe_allow_html=True)

                    with col_time:
                        st.metric("Processing Time", f"{duration:.1f}s")

                    # Decision explanation
                    st.markdown("### 📋 Decision Explanation")
                    decision_reason = result.get("decision_reason", "No reason provided.")
                    
                    if is_valid:
                        st.success(decision_reason)
                    else:
                        st.error(decision_reason)

                    # Detailed Results in Tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📋 Extracted Details",
                        "✓ Validation Checks",
                        "📜 Policy References",
                        "🔧 Raw Data",''
                        "⚡ Performance"
                    ])

                    # Tab 1 - Extracted Entities
                    with tab1:
                        st.markdown("#### Information Extracted from Claim")
                        entities = result.get("extracted_entities", {})
                        
                        if entities:
                            # Create a nice display
                            entity_data = []
                            for key, value in entities.items():
                                if value is not None and value != "":
                                    # Format the key nicely
                                    display_key = key.replace("_", " ").title()
                                    entity_data.append({"Field": display_key, "Value": value})
                            
                            if entity_data:
                                df = pd.DataFrame(entity_data)
                                st.dataframe(
                                    df, 
                                    use_container_width=True, 
                                    hide_index=True,
                                    column_config={
                                        "Field": st.column_config.TextColumn("Field", width="medium"),
                                        "Value": st.column_config.TextColumn("Value", width="large")
                                    }
                                )
                            else:
                                st.info("No entities extracted from claim.")
                        else:
                            st.warning("No entities were extracted from the claim.")

                    # Tab 2 - Validation Checks
                    with tab2:
                        st.markdown("#### Individual Validation Check Results")
                        checks = result.get("validation_checks", [])
                        
                        if not checks:
                            st.info("No validation checks were performed.")
                        else:
                            # Group by status
                            passed = [c for c in checks if c.get("status") == "PASS"]
                            failed = [c for c in checks if c.get("status") == "FAIL"]
                            na = [c for c in checks if c.get("status") == "N/A"]
                            errors = [c for c in checks if c.get("status") == "ERROR"]
                            
                            # Show metrics
                            metric_cols = st.columns(4)
                            metric_cols[0].metric("✅ Passed", len(passed))
                            metric_cols[1].metric("❌ Failed", len(failed))
                            metric_cols[2].metric("➖ Not Applicable", len(na))
                            metric_cols[3].metric("⚠️ Errors", len(errors))
                            
                            st.markdown("---")
                            
                            # Show failed checks first
                            if failed:
                                st.markdown("##### ❌ Failed Checks")
                                for check in failed:
                                    with st.expander(f"✗ {check.get('check', 'Unknown Check')}", expanded=True):
                                        st.error(f"**Detail:** {check.get('detail', 'No detail')}")
                                        st.caption(f"**Policy Reference:** {check.get('policy_reference', 'N/A')}")
                            
                            # Show passed checks
                            if passed:
                                st.markdown("##### ✅ Passed Checks")
                                for check in passed:
                                    with st.expander(f"✓ {check.get('check', 'Unknown Check')}"):
                                        st.success(f"**Detail:** {check.get('detail', 'No detail')}")
                                        st.caption(f"**Policy Reference:** {check.get('policy_reference', 'N/A')}")
                            
                            # Show N/A checks
                            if na:
                                st.markdown("##### ➖ Not Applicable")
                                for check in na:
                                    with st.expander(f"— {check.get('check', 'Unknown Check')}"):
                                        st.info(f"**Detail:** {check.get('detail', 'No detail')}")
                            
                            # Show errors
                            if errors:
                                st.markdown("##### ⚠️ Errors")
                                for check in errors:
                                    with st.expander(f"⚠ {check.get('check', 'Unknown Check')}"):
                                        st.warning(f"**Detail:** {check.get('detail', 'No detail')}")

                    # Tab 3 - Policy Clauses
                    with tab3:
                        st.markdown("#### Relevant Policy Sections Used")
                        clauses = result.get("policy_clauses_used", [])
                        
                        if not clauses:
                            st.warning("No policy clauses were referenced.")
                        else:
                            st.info(f"Retrieved {len(clauses)} relevant policy sections for validation")
                            
                            for i, clause in enumerate(clauses, 1):
                                with st.expander(f"📄 Policy Section {i}", expanded=(i <= 2)):
                                    st.text(clause)

                    # Tab 4 - Raw JSON
                    with tab4:
                        st.markdown("#### Complete Response Data")
                        st.json(result)
                        
                        # Download button
                        st.download_button(
                            label="💾 Download Result as JSON",
                            data=str(result),
                            file_name=f"validation_result_{int(time.time())}.json",
                            mime="application/json"
                        )
                    with tab5:
                        perf_metrics = result.get("performance_metrics", {})
                        if perf_metrics:
                            st.markdown("### ⚡ System Performance Metrics")
                            
                            # Display the gauges/metrics
                            m_col1, m_col2, m_col3 = st.columns(3)
                            m_col1.metric("Total Time", f"{perf_metrics.get('total_time', 0):.2f}s")
                            m_col2.metric("LLM Calls", perf_metrics.get("llm_calls", 0))
                            
                            # Add a simple bar chart of the timing breakdown
                            timing_data = {k: v for k, v in perf_metrics.items() if k.endswith('_time')}
                            if timing_data:
                                st.bar_chart(pd.Series(timing_data))
                        else:
                            st.info("Run a validation to see performance metrics here.")

            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The validation is taking too long.")
                st.info("💡 This might happen with complex policies. Try simplifying the claim or check backend logs.")
                
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend.")
                st.info("💡 Make sure the backend is running: `uvicorn api_server:app --reload`")
                
            except Exception as e:
                st.error(f"❌ Validation Error: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.exception(e)

    elif validate_button:
        st.warning("⚠️ Please enter claim details before validating.")

else:
    # No policy loaded - show instructions
    st.info("👆 Upload a policy document in the sidebar to begin.")
    
    st.markdown("---")
    st.markdown("## ✨ How This Works")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("### 1️⃣ Upload Policy")
        st.markdown("""
        - Upload any insurance policy PDF
        - System analyzes ALL requirements
        - Builds searchable knowledge base
        - Works with health, auto, home, life insurance
        """)
    
    with cols[1]:
        st.markdown("### 2️⃣ Enter Claim")
        st.markdown("""
        - Describe the claim in detail
        - Include amounts, dates, procedures
        - System extracts key information
        - Determines what needs validation
        """)
    
    with cols[2]:
        st.markdown("### 3️⃣ Get Results")
        st.markdown("""
        - AI performs comprehensive checks
        - Validates against policy terms
        - Shows pass/fail for each check
        - References specific policy clauses
        """)
    
    st.markdown("---")
    st.markdown("## 🎯 Key Features")
    
    feature_cols = st.columns(2)
    
    with feature_cols[0]:
        st.markdown("""
        **Dynamic Validation**
        - Checks adapt to your specific policy
        - Not limited to predefined rules
        - Handles unique policy clauses
        
        **Comprehensive Analysis**
        - Extracts all relevant claim details
        - Retrieves applicable policy sections
        - Performs thorough validation
        
        **Clear Explanations**
        - Shows exactly what was checked
        - References specific policy text
        - Explains pass/fail decisions
        """)
    
    with feature_cols[1]:
        st.markdown("""
        **Works With Any Policy**
        - Health insurance
        - Auto insurance
        - Home insurance
        - Life insurance
        - Commercial insurance
        
        **Powered By**
        - Local LLM (Ollama)
        - RAG (Retrieval Augmented Generation)
        - Dynamic prompt generation
        - Semantic search
        """)

# Footer
st.markdown("---")
st.caption("🤖 Powered by Ollama (Mistral) + LangChain + ChromaDB + Streamlit")


# Show system info in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.caption(f"Model: LLM via Ollama")
    st.caption(f"Embeddings: HuggingFace")
    
    if st.session_state.validation_history:
        st.caption(f"Validations: {len(st.session_state.validation_history)}")