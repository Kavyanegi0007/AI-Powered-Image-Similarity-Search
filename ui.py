import streamlit as st
import os
import json
import asyncio
import time
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

# Import your main functions (assuming they're in main2.py)
try:
    from main2 import (
        run_session, make_run_folder, ensure_remote_url, 
        download_or_copy, extract_urls_from_items, compare_all,
        AZURE_DEPLOYMENT
    )
except ImportError:
    st.error("Could not import main2.py. Please ensure the file is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Image Similarity Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #ddd;
}
.result-image {
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'run_dir' not in st.session_state:
    st.session_state.run_dir = None
if 'manifest' not in st.session_state:
    st.session_state.manifest = None

def display_image_with_score(image_path: str, score: float, source: str, col):
    """Display an image with its similarity score"""
    try:
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path, timeout=10)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_path)
        
        with col:
            st.image(img, use_column_width=True, caption=f"{source.upper()}")
            st.markdown(f"""
            <div class="metric-card">
                <h4>Similarity Score</h4>
                <h2 style="color: {'#28a745' if score >= 0.7 else '#ffc107' if score >= 0.5 else '#dc3545'}">
                    {score:.3f}
                </h2>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"File: {Path(image_path).name}")
    except Exception as e:
        col.error(f"Error loading image: {e}")

def create_score_distribution_chart(scores: Dict[str, List[Dict]]):
    """Create a distribution chart of similarity scores"""
    all_scores = []
    sources = []
    
    for source, score_list in scores.items():
        if source == 'kept':
            continue
        for item in score_list:
            all_scores.append(item['score'])
            sources.append(source.upper())
    
    if not all_scores:
        return None
    
    df = pd.DataFrame({'Score': all_scores, 'Source': sources})
    
    fig = px.histogram(
        df, x='Score', color='Source',
        title='Distribution of Similarity Scores',
        labels={'Score': 'Similarity Score', 'count': 'Number of Images'},
        opacity=0.7,
        nbins=20
    )
    
    fig.update_layout(
        showlegend=True,
        height=400,
        xaxis_title="Similarity Score",
        yaxis_title="Count"
    )
    
    return fig

def create_threshold_analysis_chart(scores: Dict[str, List[Dict]], current_threshold: float):
    """Create a chart showing how many images would be kept at different thresholds"""
    thresholds = [i/100 for i in range(0, 101, 5)]
    kept_counts = []
    
    all_items = []
    for source in ['db', 'web']:
        if source in scores:
            all_items.extend(scores[source])
    
    for threshold in thresholds:
        count = sum(1 for item in all_items if item['score'] >= threshold)
        kept_counts.append(count)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=kept_counts,
        mode='lines+markers',
        name='Images Kept',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    # Add vertical line for current threshold
    fig.add_vline(
        x=current_threshold,
        line=dict(color='red', width=2, dash='dash'),
        annotation_text=f"Current Threshold: {current_threshold}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title='Threshold Analysis: Images Kept vs Threshold Value',
        xaxis_title='Threshold Value',
        yaxis_title='Number of Images Kept',
        height=400,
        showlegend=True
    )
    
    return fig

async def run_search_async(query_image, k, threshold, azure_threshold, output_root):
    """Async wrapper for the search function"""
    try:
        # Get remote URL
        remote_url = ensure_remote_url(query_image)
        if not remote_url:
            return None, "Could not resolve image URL"
        
        # Run session
        user_text = f"Find {k} similar images for this image: {remote_url}"
        final_text, tool_payloads = await run_session(user_text)
        
        # Create run directory
        run_dir = make_run_folder(Path(output_root))
        
        # Process results
        db_results = tool_payloads.get("search_by_image", {}).get("result", []) if isinstance(tool_payloads.get("search_by_image"), dict) else tool_payloads.get("search_by_image", [])
        if isinstance(db_results, dict): 
            db_results = [db_results]
        
        # Filter DB results by Azure threshold
        filtered_db_urls = []
        for res in db_results:
            if isinstance(res, dict):
                url = res.get("decoded_path")
                score = res.get("score", 0.0)
                if url and score >= azure_threshold:
                    filtered_db_urls.append({"url": url, "score": score})
        
        # Extract web results
        web_results = extract_urls_from_items(tool_payloads.get("image_search"), source="web")
        web_urls = [r["url"] for r in web_results]
        
        # Download images
        db_dir = run_dir / "db"
        web_dir = run_dir / "web"
        
        saved_db = download_or_copy([r["url"] for r in filtered_db_urls], db_dir)
        saved_web = download_or_copy(web_urls, web_dir)
        
        # Run comparison
        scores = compare_all(
            query_url=remote_url,
            db_paths=saved_db,
            web_paths=saved_web,
            run_dir=run_dir,
            threshold=threshold
        )
        
        # Create manifest
        manifest = {
            "query_image_url": remote_url,
            "k": k,
            "run_dir": str(run_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "db_result_count": len(filtered_db_urls),
            "web_result_count": len(web_urls),
            "db_downloaded_files": [str(p) for p in saved_db],
            "web_downloaded_files": [str(p) for p in saved_web],
            "scores": scores,
            "kept_files": [r["path"] for r in scores["kept"]],
            "final_text": final_text,
            "thresholds": {
                "gpt_threshold": threshold,
                "azure_threshold": azure_threshold
            }
        }
        
        # Save manifest
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        return manifest, None
        
    except Exception as e:
        return None, str(e)

# Main UI
st.markdown('<h1 class="main-header">🔍 Image Similarity Search</h1>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Search parameters
    k = st.slider("Number of results (K)", 1, 20, 5, help="How many similar images to find")
    gpt_threshold = st.slider("GPT-4 Similarity Threshold", 0.0, 1.0, 0.5, 0.05, 
                             help="Minimum similarity score to keep images")
    azure_threshold = st.slider("Azure Cosine Threshold", 0.0, 1.0, 0.8, 0.05,
                               help="Pre-filter threshold for database results")
    
    # Output settings
    output_root = st.text_input("Output Directory", "outputs", 
                               help="Directory to save results")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        model = st.selectbox("Azure OpenAI Model", [AZURE_DEPLOYMENT], index=0)
        max_timeout = st.number_input("Request Timeout (seconds)", 10, 120, 60)

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Query Image")
    
    # Image input options
    input_method = st.radio("Choose input method:", ["Upload File", "Enter URL", "Local Path"])
    
    query_image = None
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'webp'])
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = Path("temp_query_image.jpg")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            query_image = str(temp_path)
            
            # Display uploaded image
            st.image(uploaded_file, caption="Query Image", use_column_width=True)
    
    elif input_method == "Enter URL":
        url_input = st.text_input("Image URL:", placeholder="https://example.com/image.jpg")
        if url_input:
            query_image = url_input
            try:
                st.image(url_input, caption="Query Image", use_column_width=True)
            except:
                st.warning("Could not preview image from URL")
    
    else:  # Local Path
        path_input = st.text_input("Local Path:", placeholder="path/to/image.jpg")
        if path_input and Path(path_input).exists():
            query_image = path_input
            st.image(path_input, caption="Query Image", use_column_width=True)
    
    # Search button
    if st.button("🚀 Start Search", type="primary", disabled=not query_image):
        with st.spinner("Searching for similar images..."):
            # Run the search
            try:
                manifest, error = asyncio.run(run_search_async(
                    query_image, k, gpt_threshold, azure_threshold, output_root
                ))
                
                if error:
                    st.error(f"Search failed: {error}")
                else:
                    st.session_state.manifest = manifest
                    st.session_state.run_dir = manifest["run_dir"]
                    st.success("✅ Search completed successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with col2:
    if st.session_state.manifest:
        st.subheader("📊 Search Results")
        
        manifest = st.session_state.manifest
        scores = manifest["scores"]
        
        # Summary metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("DB Images", manifest["db_result_count"])
        
        with col_b:
            st.metric("Web Images", manifest["web_result_count"])
        
        with col_c:
            st.metric("Total Processed", len(scores.get("db", [])) + len(scores.get("web", [])))
        
        with col_d:
            st.metric("Images Kept", len(scores.get("kept", [])))
        
        # Charts
        st.subheader("📈 Analysis")
        
        tab1, tab2 = st.tabs(["Score Distribution", "Threshold Analysis"])
        
        with tab1:
            fig1 = create_score_distribution_chart(scores)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No score data available for visualization")
        
        with tab2:
            fig2 = create_threshold_analysis_chart(scores, gpt_threshold)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👆 Upload an image and click 'Start Search' to begin")

# Results display
if st.session_state.manifest:
    st.subheader("🖼️ Similar Images Found")
    
    manifest = st.session_state.manifest
    scores = manifest["scores"]
    
    # Filter controls
    show_options = st.multiselect(
        "Show results from:",
        ["Database", "Web", "Kept Only"],
        default=["Kept Only"]
    )
    
    # Sort options
    sort_by = st.selectbox("Sort by:", ["Similarity Score (High to Low)", "Similarity Score (Low to High)", "Source"])
    
    # Collect and sort results
    all_results = []
    
    if "Database" in show_options and "db" in scores:
        for item in scores["db"]:
            all_results.append({**item, "display_source": "Database"})
    
    if "Web" in show_options and "web" in scores:
        for item in scores["web"]:
            all_results.append({**item, "display_source": "Web"})
    
    if "Kept Only" in show_options and "kept" in scores:
        for item in scores["kept"]:
            all_results.append({**item, "display_source": f"{item['source'].title()} (Kept)"})
    
    # Apply sorting
    if sort_by == "Similarity Score (High to Low)":
        all_results.sort(key=lambda x: x["score"], reverse=True)
    elif sort_by == "Similarity Score (Low to High)":
        all_results.sort(key=lambda x: x["score"])
    else:  # Sort by source
        all_results.sort(key=lambda x: x["display_source"])
    
    # Display results in grid
    if all_results:
        cols_per_row = 3
        for i in range(0, len(all_results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(all_results):
                    result = all_results[i + j]
                    display_image_with_score(
                        result["path"], 
                        result["score"], 
                        result["display_source"], 
                        col
                    )
    else:
        st.info("No results match the current filter criteria")
    
    # Download section
    st.subheader("💾 Download Results")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        if st.button("📄 Download Manifest JSON"):
            manifest_json = json.dumps(manifest, indent=2)
            st.download_button(
                label="💾 Save manifest.json",
                data=manifest_json,
                file_name=f"search_results_{int(time.time())}.json",
                mime="application/json"
            )
    
    with col_dl2:
        if st.button("📊 Download Scores CSV"):
            # Convert scores to DataFrame
            df_data = []
            for source in ["db", "web"]:
                if source in scores:
                    for item in scores[source]:
                        df_data.append({
                            "path": item["path"],
                            "source": source,
                            "score": item["score"],
                            "kept": item in scores.get("kept", [])
                        })
            
            if df_data:
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 Save scores.csv",
                    data=csv,
                    file_name=f"similarity_scores_{int(time.time())}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Image Similarity Search System | Powered by Azure OpenAI, Azure AI Search & SerpAPI</p>
</div>
""", unsafe_allow_html=True)