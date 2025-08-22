import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from tkinter import filedialog
import os

# ── Step 1: File selection ──
csv_paths = filedialog.askopenfilenames(
    title="Select cavity metrics CSVs",
    filetypes=[("CSV files", "*.csv")]
)
if not csv_paths:
    raise Exception("No CSVs selected.")

# ── Step 2: Color setup ──
color_palette = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"
]

# ── Step 3: Plot setup ──
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=("Depth (px)", "Width (px)", "Area (px²)")
)

summary_rows = []

for idx, path in enumerate(csv_paths):
    df = pd.read_csv(path)
    video_name = os.path.basename(path).replace("_metrics.csv", "")
    color = color_palette[idx % len(color_palette)]

    # Add plot lines
    fig.add_trace(go.Scatter(
        x=df['time_s'], y=df['depth_px'],
        mode='lines+markers',
        name=video_name,
        legendgroup=video_name,
        line=dict(color=color),
        marker=dict(color=color),
        hovertemplate=f"<b>{video_name}</b><br>Time: %{{x:.2f}}s<br>Depth: %{{y}}px",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['time_s'], y=df['width_px'],
        mode='lines+markers',
        name=video_name,
        legendgroup=video_name,
        showlegend=False,
        line=dict(color=color),
        marker=dict(color=color),
        hovertemplate=f"<b>{video_name}</b><br>Time: %{{x:.2f}}s<br>Width: %{{y}}px",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df['time_s'], y=df['area_px2'],
        mode='lines+markers',
        name=video_name,
        legendgroup=video_name,
        showlegend=False,
        line=dict(color=color),
        marker=dict(color=color),
        hovertemplate=f"<b>{video_name}</b><br>Time: %{{x:.2f}}s<br>Area: %{{y}}px²",
    ), row=3, col=1)

    # Summary data
    max_depth_idx = df['depth_px'].idxmax()
    max_width_idx = df['width_px'].idxmax()
    max_area_idx = df['area_px2'].idxmax()

    summary_rows.append({
        "Video": video_name,
        "Max Depth": df.at[max_depth_idx, 'depth_px'],
        "Time (Depth)": round(df.at[max_depth_idx, 'time_s'], 2),
        "Frame (Depth)": df.at[max_depth_idx, 'frame'] if 'frame' in df.columns else max_depth_idx,

        "Max Width": df.at[max_width_idx, 'width_px'],
        "Time (Width)": round(df.at[max_width_idx, 'time_s'], 2),
        "Frame (Width)": df.at[max_width_idx, 'frame'] if 'frame' in df.columns else max_width_idx,

        "Max Area": df.at[max_area_idx, 'area_px2'],
        "Time (Area)": round(df.at[max_area_idx, 'time_s'], 2),
        "Frame (Area)": df.at[max_area_idx, 'frame'] if 'frame' in df.columns else max_area_idx,
    })

# ── Step 4: Format Plot Layout ──
fig.update_layout(
    height=900,
    title_text="Cavity Metrics Over Time",
    hovermode="x unified",
    template="plotly_white",
    legend_title="Video",
    font=dict(size=14)
)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)

# ── Step 5: Convert summary to HTML table ──
summary_df = pd.DataFrame(summary_rows)

summary_html = summary_df.to_html(
    index=False,
    classes="summary-table",
    border=0,
    justify="center"
)

# ── Step 6: Inject into HTML page ──
html_path = "cavity_metrics_summary.html"
fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

# Wrap with a simple HTML document
full_html = f"""
<html>
<head>
    <title>Cavity Metrics Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
        }}
        .summary-table {{
            margin-top: 50px;
            width: 100%;
            border-collapse: collapse;
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #ccc;
            padding: 8px 12px;
            text-align: center;
        }}
        .summary-table th {{
            background-color: #f2f2f2;
        }}
        h2 {{
            margin-top: 60px;
        }}
    </style>
</head>
<body>
    {fig_html}
    <h2> Comparison Summary</h2>
    {summary_html}
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

# Open in browser
import webbrowser
webbrowser.open(html_path)
