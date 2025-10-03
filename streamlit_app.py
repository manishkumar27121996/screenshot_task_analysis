# import streamlit as st
# import openai
# import base64
# import json
# import os
# from dotenv import load_dotenv
# import pandas as pd
# from datetime import datetime
# import zipfile
# from io import BytesIO

# # Load environment variables
# load_dotenv()

# # Set page config
# st.set_page_config(
#     page_title="Screenshot Task Analyzer",
#     page_icon="🔍",
#     layout="wide"
# )

# # Initialize OpenAI API key
# openai.api_key = os.getenv('OPENAI_API_KEY')

# def analyze_screenshot(image_bytes, image_name):
#     """
#     Analyze a screenshot to determine what task the user is working on
    
#     Args:
#         image_bytes: Image file bytes
#         image_name: Name of the image file
    
#     Returns:
#         Dictionary with detailed task analysis
#     """
#     try:
#         # Encode image to base64
#         image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
#         # Determine image format from filename
#         image_format = image_name.lower().split('.')[-1]
#         if image_format == 'jpg':
#             image_format = 'jpeg'
        
#         # Prepare the messages with image
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/{image_format};base64,{image_base64}"
#                         }
#                     },
#                     {
#                         "type": "text",
#                         "text": """Analyze this screenshot and determine what task the user is working on.

# Provide analysis in this JSON format:
# {
#     "primary_task": "Main task being performed",
#     "application": "Software/tool being used",
#     "task_category": "coding/writing/design/browsing/communication/other",
#     "estimated_complexity": "simple/moderate/complex",
#     "work_type": "professional/personal/educational",
#     "visible_content_summary": "What's visible on screen",
#     "inferred_goal": "What the user is trying to accomplish"
# }

# Return only valid JSON."""
#                     }
#                 ]
#             }
#         ]
        
#         # Make the API call
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages,
#             max_tokens=1000,
#             temperature=0.7
#         )
        
#         result_text = response.choices[0].message.content
        
#         # Parse JSON from response
#         start_idx = result_text.find('{')
#         end_idx = result_text.rfind('}') + 1
#         json_str = result_text[start_idx:end_idx]
#         task_analysis = json.loads(json_str)
        
#         # Add image name to result
#         task_analysis['image_name'] = image_name
#         task_analysis['status'] = 'success'
        
#         return task_analysis
        
#     except Exception as e:
#         return {
#             "image_name": image_name,
#             "status": "error",
#             "error": str(e)
#         }

# def create_results_dataframe(results):
#     """Convert results to pandas DataFrame"""
#     df_data = []
#     for result in results:
#         if result['status'] == 'success':
#             df_data.append({
#                 'Image Name': result['image_name'],
#                 'Primary Task': result.get('primary_task', 'N/A'),
#                 'Application': result.get('application', 'N/A'),
#                 'Category': result.get('task_category', 'N/A'),
#                 'Complexity': result.get('estimated_complexity', 'N/A'),
#                 'Work Type': result.get('work_type', 'N/A'),
#                 'Content Summary': result.get('visible_content_summary', 'N/A'),
#                 'Inferred Goal': result.get('inferred_goal', 'N/A')
#             })
#         else:
#             df_data.append({
#                 'Image Name': result['image_name'],
#                 'Primary Task': 'ERROR',
#                 'Application': result.get('error', 'Unknown error'),
#                 'Category': '-',
#                 'Complexity': '-',
#                 'Work Type': '-',
#                 'Content Summary': '-',
#                 'Inferred Goal': '-'
#             })
    
#     return pd.DataFrame(df_data)

# def create_download_zip(results):
#     """Create a ZIP file with JSON results"""
#     zip_buffer = BytesIO()
#     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
#         # Add individual JSON files
#         for result in results:
#             json_filename = f"{result['image_name'].split('.')[0]}_analysis.json"
#             json_content = json.dumps(result, indent=2)
#             zip_file.writestr(json_filename, json_content)
        
#         # Add combined results
#         combined_json = json.dumps(results, indent=2)
#         zip_file.writestr('all_results.json', combined_json)
    
#     zip_buffer.seek(0)
#     return zip_buffer

# # Main App
# def main():
#     # Header
#     st.title("🔍 Screenshot Task Analyzer")
#     st.markdown("Upload multiple screenshots to analyze what tasks users are working on")
    
#     # Sidebar for settings
#     with st.sidebar:
#         st.header("⚙️ Settings")
#         st.info("**API:** OpenAI GPT-4o-mini")
        
#         if not openai.api_key:
#             st.error("⚠️ OPENAI_API_KEY not found in environment variables")
#             st.stop()
#         else:
#             st.success("✅ API Key loaded")
        
#         st.markdown("---")
#         st.markdown("### 📊 Cost Estimation")
#         st.markdown("**~$0.0001 per screenshot**")
#         st.markdown("~10,000 images per $1")
        
#         st.markdown("---")
#         st.markdown("### 📝 Supported Formats")
#         st.markdown("- PNG\n- JPG/JPEG\n- WebP\n- GIF")
    
#     # File uploader
#     uploaded_files = st.file_uploader(
#         "Upload Screenshots",
#         type=['png', 'jpg', 'jpeg', 'webp', 'gif'],
#         accept_multiple_files=True,
#         help="Select one or more screenshot images to analyze"
#     )
    
#     if uploaded_files:
#         st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
#         # Preview uploaded images
#         with st.expander("📸 Preview Uploaded Images", expanded=False):
#             cols = st.columns(4)
#             for idx, file in enumerate(uploaded_files):
#                 with cols[idx % 4]:
#                     # st.image(file, caption=file.name, use_container_width=True)
#                     st.image(file, caption=file.name, width='stretch')
        
#         # Analyze button
#         # if st.button("🚀 Analyze All Screenshots", type="primary", use_container_width=True):
#         if st.button("🚀 Analyze All Screenshots", type="primary", width='stretch'):
#             results = []
            
#             # Progress bar
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             # Process each image
#             for idx, uploaded_file in enumerate(uploaded_files):
#                 status_text.text(f"Analyzing {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})")
                
#                 # Read image bytes
#                 image_bytes = uploaded_file.read()
                
#                 # Analyze
#                 result = analyze_screenshot(image_bytes, uploaded_file.name)
#                 results.append(result)
                
#                 # Update progress
#                 progress_bar.progress((idx + 1) / len(uploaded_files))
                
#                 # Reset file pointer for potential re-reading
#                 uploaded_file.seek(0)
            
#             status_text.text("✅ Analysis complete!")
#             progress_bar.empty()
            
#             # Store results in session state
#             st.session_state['results'] = results
#             st.session_state['analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     # Display results if available
#     if 'results' in st.session_state:
#         st.markdown("---")
#         st.header("📊 Analysis Results")
#         st.caption(f"Analyzed at: {st.session_state['analysis_timestamp']}")
        
#         results = st.session_state['results']
        
#         # Summary statistics
#         col1, col2, col3, col4 = st.columns(4)
        
#         successful = sum(1 for r in results if r['status'] == 'success')
#         failed = len(results) - successful
        
#         with col1:
#             st.metric("Total Images", len(results))
#         with col2:
#             st.metric("Successful", successful)
#         with col3:
#             st.metric("Failed", failed)
#         with col4:
#             estimated_cost = len(results) * 0.0001
#             st.metric("Est. Cost", f"${estimated_cost:.4f}")
        
#         # Tabs for different views
#         tab1, tab2, tab3 = st.tabs(["📋 Table View", "🔍 Detailed View", "📥 Export"])
        
#         with tab1:
#             # Create and display DataFrame
#             df = create_results_dataframe(results)
#             st.dataframe(
#                 df,
#                 # use_container_width=True,
#                 width='stretch',
#                 hide_index=True,
#                 column_config={
#                     "Image Name": st.column_config.TextColumn("Image Name", width="medium"),
#                     "Primary Task": st.column_config.TextColumn("Primary Task", width="large"),
#                     "Content Summary": st.column_config.TextColumn("Content Summary", width="large"),
#                 }
#             )
            
#             # Download CSV
#             csv = df.to_csv(index=False)
#             st.download_button(
#                 label="📥 Download as CSV",
#                 data=csv,
#                 file_name=f"task_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                 mime="text/csv"
#             )
        
#         with tab2:
#             # Detailed view for each image
#             for result in results:
#                 with st.expander(f"🖼️ {result['image_name']}", expanded=False):
#                     if result['status'] == 'success':
#                         col1, col2 = st.columns(2)
                        
#                         with col1:
#                             st.markdown("**📋 Primary Task:**")
#                             st.info(result.get('primary_task', 'N/A'))
                            
#                             st.markdown("**💻 Application:**")
#                             st.info(result.get('application', 'N/A'))
                            
#                             st.markdown("**🏷️ Task Category:**")
#                             st.info(result.get('task_category', 'N/A'))
                        
#                         with col2:
#                             st.markdown("**⚙️ Complexity Level:**")
#                             st.info(result.get('estimated_complexity', 'N/A'))
                            
#                             st.markdown("**🎯 Work Type:**")
#                             st.info(result.get('work_type', 'N/A'))
                        
#                         st.markdown("**👁️ Visible Content:**")
#                         st.write(result.get('visible_content_summary', 'N/A'))
                        
#                         st.markdown("**🎯 Inferred Goal:**")
#                         st.write(result.get('inferred_goal', 'N/A'))
                        
#                         # Show raw JSON
#                         with st.expander("📄 View Raw JSON"):
#                             st.json(result)
#                     else:
#                         st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        
#         with tab3:
#             st.markdown("### 📥 Export Options")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 # Download all as JSON
#                 all_json = json.dumps(results, indent=2)
#                 st.download_button(
#                     label="📄 Download All as JSON",
#                     data=all_json,
#                     file_name=f"all_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                     mime="application/json"
#                 )
            
#             with col2:
#                 # Download as ZIP with individual files
#                 zip_buffer = create_download_zip(results)
#                 st.download_button(
#                     label="🗜️ Download as ZIP (Individual JSONs)",
#                     data=zip_buffer,
#                     file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
#                     mime="application/zip"
#                 )

# if __name__ == "__main__":
#     main()





import streamlit as st
import openai
import base64
import json
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import zipfile
from io import BytesIO

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Screenshot Task Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define known projects
KNOWN_PROJECTS = [
    "Ralvie AI",
    "Sundial",
    "Cirrus Time",
    "Easy Time Pro"
]

def analyze_screenshot(image_bytes, image_name):
    """
    Analyze a screenshot to determine what task the user is working on
    
    Args:
        image_bytes: Image file bytes
        image_name: Name of the image file
    
    Returns:
        Dictionary with detailed task analysis
    """
    try:
        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Determine image format from filename
        image_format = image_name.lower().split('.')[-1]
        if image_format == 'jpg':
            image_format = 'jpeg'
        
        # Prepare the messages with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"""Analyze this screenshot and determine what task the user is working on.

IMPORTANT: Look for any project names or partial project names in the screenshot. The known projects are:
- Ralvie AI (also look for: Ralvie, ralvie, RALVIE)
- Sundial (also look for: sundial, SUNDIAL)
- Cirrus Time (also look for: Cirrus, cirrus, CIRRUS, Cirrus Time, cirrustime)
- Easy Time Pro (also look for: Easy Time, EasyTime, easytime, easy time pro, easytimepro)

If you find any of these project names or their variations, match them to the correct project name from the list above.
If the project name is not in the list or cannot be found, set "project_name" to "Project not listed".

Provide analysis in this JSON format:
{{
    "project_name": "Exact project name from the list above, or 'Project not listed'",
    "primary_task": "Main task being performed",
    "application": "Software/tool being used",
    "task_category": "coding/writing/design/browsing/communication/other",
    "estimated_complexity": "simple/moderate/complex",
    "work_type": "professional/personal/educational",
    "visible_content_summary": "What's visible on screen",
    "inferred_goal": "What the user is trying to accomplish"
}}

Return only valid JSON."""
                    }
                ]
            }
        ]
        
        # Make the API call
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        result_text = response.choices[0].message.content
        
        # Parse JSON from response
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}') + 1
        json_str = result_text[start_idx:end_idx]
        task_analysis = json.loads(json_str)
        
        # Add image name to result
        task_analysis['image_name'] = image_name
        task_analysis['status'] = 'success'
        
        return task_analysis
        
    except Exception as e:
        return {
            "image_name": image_name,
            "status": "error",
            "error": str(e)
        }

def create_results_dataframe(results):
    """Convert results to pandas DataFrame"""
    df_data = []
    for result in results:
        if result['status'] == 'success':
            df_data.append({
                'Image Name': result['image_name'],
                'Project': result.get('project_name', 'Project not listed'),
                'Primary Task': result.get('primary_task', 'N/A'),
                'Application': result.get('application', 'N/A'),
                'Category': result.get('task_category', 'N/A'),
                'Complexity': result.get('estimated_complexity', 'N/A'),
                'Work Type': result.get('work_type', 'N/A'),
                'Content Summary': result.get('visible_content_summary', 'N/A'),
                'Inferred Goal': result.get('inferred_goal', 'N/A')
            })
        else:
            df_data.append({
                'Image Name': result['image_name'],
                'Project': '-',
                'Primary Task': 'ERROR',
                'Application': result.get('error', 'Unknown error'),
                'Category': '-',
                'Complexity': '-',
                'Work Type': '-',
                'Content Summary': '-',
                'Inferred Goal': '-'
            })
    
    return pd.DataFrame(df_data)

def create_download_zip(results):
    """Create a ZIP file with JSON results"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add individual JSON files
        for result in results:
            json_filename = f"{result['image_name'].split('.')[0]}_analysis.json"
            json_content = json.dumps(result, indent=2)
            zip_file.writestr(json_filename, json_content)
        
        # Add combined results
        combined_json = json.dumps(results, indent=2)
        zip_file.writestr('all_results.json', combined_json)
    
    zip_buffer.seek(0)
    return zip_buffer

# Main App
def main():
    # Header
    st.title("🔍 Screenshot Task Analyzer")
    st.markdown("Upload multiple screenshots to analyze what tasks users are working on")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("**API:** OpenAI GPT-4o-mini")
        
        if not openai.api_key:
            st.error("⚠️ OPENAI_API_KEY not found in environment variables")
            st.stop()
        else:
            st.success("✅ API Key loaded")
        
        st.markdown("---")
        st.markdown("### 📂 Known Projects")
        for project in KNOWN_PROJECTS:
            st.markdown(f"- {project}")
        
        st.markdown("---")
        st.markdown("### 📝 Supported Formats")
        st.markdown("- PNG\n- JPG/JPEG\n- WebP\n- GIF")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Screenshots",
        type=['png', 'jpg', 'jpeg', 'webp', 'gif'],
        accept_multiple_files=True,
        help="Select one or more screenshot images to analyze"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        # Preview uploaded images
        with st.expander("📸 Preview Uploaded Images", expanded=False):
            cols = st.columns(4)
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 4]:
                    st.image(file, caption=file.name, width='stretch')
        
        # Analyze button
        if st.button("🚀 Analyze All Screenshots", type="primary", width='stretch'):
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each image
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Analyzing {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})")
                
                # Read image bytes
                image_bytes = uploaded_file.read()
                
                # Analyze
                result = analyze_screenshot(image_bytes, uploaded_file.name)
                results.append(result)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Reset file pointer for potential re-reading
                uploaded_file.seek(0)
            
            status_text.text("✅ Analysis complete!")
            progress_bar.empty()
            
            # Store results in session state
            st.session_state['results'] = results
            st.session_state['analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display results if available
    if 'results' in st.session_state:
        st.markdown("---")
        st.header("📊 Analysis Results")
        st.caption(f"Analyzed at: {st.session_state['analysis_timestamp']}")
        
        results = st.session_state['results']
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        
        with col1:
            st.metric("Total Images", len(results))
        with col2:
            st.metric("Successful", successful)
        with col3:
            st.metric("Failed", failed)
        
        # Project distribution
        if successful > 0:
            st.markdown("### 📂 Project Distribution")
            project_counts = {}
            for r in results:
                if r['status'] == 'success':
                    project = r.get('project_name', 'Project not listed')
                    project_counts[project] = project_counts.get(project, 0) + 1
            
            # Display as columns
            proj_cols = st.columns(len(project_counts))
            for idx, (project, count) in enumerate(project_counts.items()):
                with proj_cols[idx]:
                    st.metric(project, count)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Table View", "🔍 Detailed View", "📥 Export"])
        
        with tab1:
            # Create and display DataFrame
            df = create_results_dataframe(results)
            st.dataframe(
                df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Image Name": st.column_config.TextColumn("Image Name", width="medium"),
                    "Project": st.column_config.TextColumn("Project", width="medium"),
                    "Primary Task": st.column_config.TextColumn("Primary Task", width="large"),
                    "Content Summary": st.column_config.TextColumn("Content Summary", width="large"),
                }
            )
            
            # Filter by project
            st.markdown("### 🔍 Filter by Project")
            selected_project = st.selectbox(
                "Select Project",
                ["All"] + list(set(df['Project'].tolist()))
            )
            
            if selected_project != "All":
                filtered_df = df[df['Project'] == selected_project]
                st.dataframe(filtered_df, width='stretch', hide_index=True)
            
            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"task_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab2:
            # Detailed view for each image
            for result in results:
                with st.expander(f"🖼️ {result['image_name']}", expanded=False):
                    if result['status'] == 'success':
                        # Project highlight
                        project = result.get('project_name', 'Project not listed')
                        if project != 'Project not listed':
                            st.success(f"📂 **Project:** {project}")
                        else:
                            st.warning(f"📂 **Project:** {project}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📋 Primary Task:**")
                            st.info(result.get('primary_task', 'N/A'))
                            
                            st.markdown("**💻 Application:**")
                            st.info(result.get('application', 'N/A'))
                            
                            st.markdown("**🏷️ Task Category:**")
                            st.info(result.get('task_category', 'N/A'))
                        
                        with col2:
                            st.markdown("**⚙️ Complexity Level:**")
                            st.info(result.get('estimated_complexity', 'N/A'))
                            
                            st.markdown("**🎯 Work Type:**")
                            st.info(result.get('work_type', 'N/A'))
                        
                        st.markdown("**👁️ Visible Content:**")
                        st.write(result.get('visible_content_summary', 'N/A'))
                        
                        st.markdown("**🎯 Inferred Goal:**")
                        st.write(result.get('inferred_goal', 'N/A'))
                        
                        # Show raw JSON
                        with st.expander("📄 View Raw JSON"):
                            st.json(result)
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        with tab3:
            st.markdown("### 📥 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download all as JSON
                all_json = json.dumps(results, indent=2)
                st.download_button(
                    label="📄 Download All as JSON",
                    data=all_json,
                    file_name=f"all_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Download as ZIP with individual files
                zip_buffer = create_download_zip(results)
                st.download_button(
                    label="🗜️ Download as ZIP (Individual JSONs)",
                    data=zip_buffer,
                    file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()