import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import numpy as np

# Page Configuration
st.set_page_config(
    layout="wide",
    page_title="Performance Metrics Dashboard",
    page_icon="📊"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stat-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "metrics_data" not in st.session_state:
    st.session_state.metrics_data = []
if "current_metric" not in st.session_state:
    st.session_state.current_metric = None

# Header
st.markdown('<div class="main-header">📊 Performance Metrics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time RAG Pipeline Performance Analysis</div>', unsafe_allow_html=True)

# Sidebar - Data Input
st.sidebar.markdown("## 📥 Load Performance Data")

input_method = st.sidebar.radio(
    "Choose Input Method:",
    ["Upload JSON File", "Paste JSON Data", "Load from History"]
)

metrics = None

if input_method == "Upload JSON File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload validation result JSON:",
        type=["json"],
        help="Upload a JSON file containing validation results with performance_metrics"
    )
    
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            if "performance_metrics" in data:
                metrics = data["performance_metrics"]
                st.session_state.current_metric = metrics
                st.sidebar.success("✅ Metrics loaded successfully!")
            else:
                st.sidebar.error("❌ No performance_metrics found in JSON")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")

elif input_method == "Paste JSON Data":
    json_input = st.sidebar.text_area(
        "Paste JSON data here:",
        height=200,
        placeholder='{"performance_metrics": {...}}'
    )
    
    if st.sidebar.button("Load Data"):
        try:
            data = json.loads(json_input)
            if "performance_metrics" in data:
                metrics = data["performance_metrics"]
                st.session_state.current_metric = metrics
                st.sidebar.success("✅ Metrics loaded successfully!")
            else:
                st.sidebar.error("❌ No performance_metrics found in JSON")
        except json.JSONDecodeError:
            st.sidebar.error("❌ Invalid JSON format")

elif input_method == "Load from History":
    log_dir = "performance_logs"
    if os.path.exists(log_dir):
        files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        if files:
            # Sort by time (newest first)
            files.sort(reverse=True)
            selected_file = st.sidebar.selectbox("Select Run:", files)
            
            with open(os.path.join(log_dir, selected_file), "r") as f:
                data = json.load(f)
                metrics = data.get("performance_metrics")
                st.session_state.current_metric = metrics

# Save current metrics to history
if metrics and st.sidebar.button("💾 Save to History"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.metrics_data.append({
        "timestamp": timestamp,
        "metrics": metrics
    })
    st.sidebar.success(f"✅ Saved as Run {len(st.session_state.metrics_data)}")

# Clear history
if st.session_state.metrics_data and st.sidebar.button("🗑️ Clear History"):
    st.session_state.metrics_data = []
    st.sidebar.success("History cleared")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Sample JSON Structure")
st.sidebar.code("""
{
  "performance_metrics": {
    "total_time": 12.5,
    "entity_extraction_time": 2.3,
    "retrieval_time": 1.8,
    "parallel_checks_time": 7.2,
    "llm_calls": 5,
    "timeouts": 0
  }
}
""", language="json")

# Main Content
if metrics:
    # Extract timing data
    timing_data = {}
    other_metrics = {}
    
    for key, value in metrics.items():
        if key.endswith('_time') and isinstance(value, (int, float)):
            label = key.replace('_time', '').replace('_', ' ').title()
            timing_data[label] = value
        elif isinstance(value, (int, float)):
            label = key.replace('_', ' ').title()
            other_metrics[label] = value
    
    total_time = metrics.get('total_time', sum(timing_data.values()))
    
    # Key Metrics Row
    st.markdown("## 🎯 Key Performance Indicators")
    
    kpi_cols = st.columns(5)
    
    with kpi_cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⏱️ Total Time", f"{total_time:.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with kpi_cols[1]:
        llm_calls = metrics.get('llm_calls', 0)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🤖 LLM Calls", llm_calls)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with kpi_cols[2]:
        avg_time = (total_time / llm_calls) if llm_calls > 0 else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚡ Avg Call Time", f"{avg_time:.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with kpi_cols[3]:
        timeouts = metrics.get('timeouts', 0)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⏰ Timeouts", timeouts)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with kpi_cols[4]:
        cache_hits = metrics.get('cache_hits', 0)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💾 Cache Hits", cache_hits)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualization Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⏱️ Time Distribution",
        "📊 Detailed Breakdown", 
        "🔄 Pipeline Flow",
        "📈 Metrics Analysis",
        "📋 Data Table"
    ])
    
    # Tab 1: Time Distribution (Pie + Donut)
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🥧 Time Distribution by Stage")
            if timing_data:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(timing_data.keys()),
                    values=list(timing_data.values()),
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Set3,
                    textinfo='label+percent',
                    textposition='outside',
                    hovertemplate='<b>%{label}</b><br>Time: %{value:.2f}s<br>Percentage: %{percent}<extra></extra>'
                )])
                fig_pie.update_layout(
                    height=500,
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No timing data available")
        
        with col2:
            st.markdown("### 📊 Stage Comparison")
            if timing_data:
                # Sort by time
                sorted_stages = dict(sorted(timing_data.items(), key=lambda x: x[1], reverse=True))
                
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=list(sorted_stages.values()),
                        y=list(sorted_stages.keys()),
                        orientation='h',
                        marker=dict(
                            color=list(sorted_stages.values()),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Time (s)")
                        ),
                        text=[f"{v:.2f}s" for v in sorted_stages.values()],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Time: %{x:.2f}s<extra></extra>'
                    )
                ])
                fig_bar.update_layout(
                    height=500,
                    xaxis_title="Time (seconds)",
                    yaxis_title="Stage",
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # Tab 2: Detailed Breakdown
    with tab2:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📉 Waterfall Chart - Sequential Time Flow")
            if timing_data:
                stages = list(timing_data.keys())
                times = list(timing_data.values())
                
                fig_waterfall = go.Figure(go.Waterfall(
                    name="Pipeline Stages",
                    orientation="v",
                    measure=["relative"] * len(stages) + ["total"],
                    x=stages + ["Total"],
                    y=times + [sum(times)],
                    text=[f"{t:.2f}s" for t in times] + [f"{sum(times):.2f}s"],
                    textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#667eea"}},
                    totals={"marker": {"color": "#764ba2"}},
                    hovertemplate='<b>%{x}</b><br>Time: %{y:.2f}s<extra></extra>'
                ))
                fig_waterfall.update_layout(
                    height=500,
                    yaxis_title="Time (seconds)",
                    showlegend=False
                )
                st.plotly_chart(fig_waterfall, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Time Percentages")
            if timing_data and total_time > 0:
                for stage, time in timing_data.items():
                    percentage = (time / total_time) * 100
                    st.markdown(f"""
                    <div class="stat-box">
                        <strong>{stage}</strong><br>
                        <span style="font-size: 1.2em;">{time:.2f}s</span> 
                        <span style="color: #667eea; font-weight: bold;">({percentage:.1f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Tab 3: Pipeline Flow
    with tab3:
        st.markdown("### 🔄 Pipeline Execution Timeline")
        
        if timing_data:
            # Create a Gantt-like chart
            stages = list(timing_data.keys())
            times = list(timing_data.values())
            
            # Calculate cumulative start times
            start_times = [0]
            for i in range(len(times) - 1):
                start_times.append(start_times[-1] + times[i])
            
            fig_timeline = go.Figure()
            
            colors = px.colors.qualitative.Set3[:len(stages)]
            
            for i, (stage, time, start, color) in enumerate(zip(stages, times, start_times, colors)):
                fig_timeline.add_trace(go.Bar(
                    name=stage,
                    x=[time],
                    y=[stage],
                    orientation='h',
                    marker=dict(color=color),
                    text=f"{time:.2f}s",
                    textposition='inside',
                    hovertemplate=f'<b>{stage}</b><br>Duration: {time:.2f}s<br>Start: {start:.2f}s<br>End: {start+time:.2f}s<extra></extra>',
                    base=start
                ))
            
            fig_timeline.update_layout(
                height=400,
                xaxis_title="Time (seconds)",
                yaxis_title="Pipeline Stage",
                showlegend=False,
                barmode='overlay'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Show timing efficiency
            st.markdown("### ⚡ Efficiency Metrics")
            efficiency_cols = st.columns(3)
            
            if timing_data:
                slowest_stage = max(timing_data.items(), key=lambda x: x[1])
                fastest_stage = min(timing_data.items(), key=lambda x: x[1])
                
                with efficiency_cols[0]:
                    st.metric(
                        "🐌 Slowest Stage",
                        slowest_stage[0],
                        f"{slowest_stage[1]:.2f}s"
                    )
                
                with efficiency_cols[1]:
                    st.metric(
                        "⚡ Fastest Stage",
                        fastest_stage[0],
                        f"{fastest_stage[1]:.2f}s"
                    )
                
                with efficiency_cols[2]:
                    avg_stage_time = np.mean(list(timing_data.values()))
                    st.metric(
                        "📊 Average Stage Time",
                        f"{avg_stage_time:.2f}s"
                    )
    
    # Tab 4: Metrics Analysis
    with tab4:
        st.markdown("### 📈 System Metrics Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # LLM Performance
            st.markdown("#### 🤖 LLM Performance")
            llm_metrics = {
                'LLM Calls': metrics.get('llm_calls', 0),
                'Timeouts': metrics.get('timeouts', 0),
                'Avg Time per Call': avg_time
            }
            
            fig_llm = go.Figure(data=[
                go.Bar(
                    x=list(llm_metrics.keys()),
                    y=list(llm_metrics.values()),
                    marker_color=['#3498db', '#e74c3c', '#2ecc71'],
                    text=[f"{v:.2f}" for v in llm_metrics.values()],
                    textposition='outside'
                )
            ])
            fig_llm.update_layout(height=400, showlegend=False, yaxis_title="Value")
            st.plotly_chart(fig_llm, use_container_width=True)
        
        with col2:
            # Cache Performance
            st.markdown("#### 💾 Cache Performance")
            cache_hits = metrics.get('cache_hits', 0)
            cache_misses = metrics.get('cache_misses', 0)
            total_requests = cache_hits + cache_misses
            
            if total_requests > 0:
                hit_rate = (cache_hits / total_requests) * 100
                
                fig_cache = go.Figure(data=[go.Pie(
                    labels=['Cache Hits', 'Cache Misses'],
                    values=[cache_hits, cache_misses],
                    marker_colors=['#2ecc71', '#e74c3c'],
                    hole=0.5,
                    textinfo='label+value+percent'
                )])
                fig_cache.update_layout(
                    height=400,
                    annotations=[dict(text=f'{hit_rate:.1f}%<br>Hit Rate', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig_cache, use_container_width=True)
            else:
                st.info("No cache data available")
        
        # All metrics table
        st.markdown("#### 📋 Complete Metrics Summary")
        all_metrics_data = []
        for key, value in metrics.items():
            metric_name = key.replace('_', ' ').title()
            if isinstance(value, float):
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
            all_metrics_data.append({"Metric": metric_name, "Value": formatted_value})
        
        df_all_metrics = pd.DataFrame(all_metrics_data)
        st.dataframe(df_all_metrics, use_container_width=True, hide_index=True)
    
    # Tab 5: Data Table
    with tab5:
        st.markdown("### 📋 Raw Performance Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⏱️ Timing Data")
            if timing_data:
                timing_df = pd.DataFrame([
                    {
                        "Stage": stage,
                        "Time (s)": f"{time:.3f}",
                        "Percentage": f"{(time/total_time*100):.1f}%" if total_time > 0 else "N/A"
                    }
                    for stage, time in timing_data.items()
                ])
                st.dataframe(timing_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 📊 Other Metrics")
            if other_metrics:
                other_df = pd.DataFrame([
                    {"Metric": metric, "Value": value}
                    for metric, value in other_metrics.items()
                ])
                st.dataframe(other_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### 📥 Export Data")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Export as CSV
            export_df = pd.DataFrame([metrics])
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with export_col2:
            # Export as JSON
            json_str = json.dumps(metrics, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Show raw JSON
        with st.expander("🔍 View Raw JSON"):
            st.json(metrics)

else:
    # No data loaded - show instructions
    st.info("👈 Load performance data from the sidebar to begin analysis")
    
    st.markdown("---")
    st.markdown("## 📊 Dashboard Features")
    
    features = st.columns(3)
    
    with features[0]:
        st.markdown("""
        ### ⏱️ Time Analysis
        - Interactive pie charts
        - Waterfall diagrams
        - Stage-by-stage breakdown
        - Percentage calculations
        """)
    
    with features[1]:
        st.markdown("""
        ### 📈 Performance Metrics
        - LLM call tracking
        - Timeout monitoring
        - Cache hit rates
        - Efficiency metrics
        """)
    
    with features[2]:
        st.markdown("""
        ### 💾 Data Management
        - Save to history
        - Compare runs
        - Export CSV/JSON
        - Raw data access
        """)
    
    st.markdown("---")
    st.markdown("## 🎯 How to Use")
    
    st.markdown("""
    1. **Load Data**: Upload a JSON file, paste JSON data, or load from history
    2. **Analyze**: View interactive visualizations across different tabs
    3. **Identify Bottlenecks**: See which stages take the most time
    4. **Export**: Download data in CSV or JSON format for further analysis
    5. **Compare**: Save multiple runs to history and compare performance
    """)

# Footer
st.markdown("---")
st.caption("📊 Performance Metrics Dashboard | Built with Streamlit + Plotly")

if st.session_state.metrics_data:
    st.caption(f"💾 {len(st.session_state.metrics_data)} runs saved in history")