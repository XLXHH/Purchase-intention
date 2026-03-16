import time
import threading
import io
import queue
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from Ana import run_pipeline
import os

DEFAULT_EVENTS_PATH = "data/new.xlsx"
st.set_page_config(
    page_title="电商购买意向预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_custom_css():
    st.markdown("""
    <style>
    :root {
    --bg-1: #f6fbff;
    --bg-2: #eef6ff;
    --bg-3: #dceeff;

    --card-bg: rgba(255, 255, 255, 0.78);
    --card-border: rgba(96, 165, 250, 0.18);

    --text-main: #0f172a;
    --text-soft: #334155;
    --text-muted: #64748b;

    --primary: #2563eb;
    --primary-2: #60a5fa;
    --accent: #38bdf8;

    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;

    --shadow: 0 18px 40px rgba(37, 99, 235, 0.10);
    }

    .stApp {
    background:
        radial-gradient(circle at 12% 18%, rgba(96,165,250,0.22), transparent 26%),
        radial-gradient(circle at 88% 12%, rgba(56,189,248,0.18), transparent 24%),
        radial-gradient(circle at 50% 85%, rgba(191,219,254,0.35), transparent 28%),
        linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 50%, var(--bg-3) 100%);
    color: var(--text-main);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(219,234,254,0.88));
    border: 1px solid rgba(96,165,250,0.20);
    border-radius: 28px;
    padding: 30px 34px;
    margin-bottom: 22px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(12px);
    }
    
    .hero-card::before {
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: rgba(96,165,250,0.16);
        filter: blur(24px);
    }

    .hero-title {
    font-size: 36px;
    font-weight: 800;
    color: var(--text-main);
    margin-bottom: 8px;
    letter-spacing: 0.3px;
    }
    
    .hero-subtitle {
        font-size: 15px;
        color: var(--text-soft);
        line-height: 1.8;
    }

    .glass-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 22px;
    padding: 18px 20px;
    box-shadow: 0 10px 30px rgba(37,99,235,0.08);
    backdrop-filter: blur(10px);
    margin-bottom: 16px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 800;
        color: var(--text-main);
        margin: 8px 0 12px 0;
    }

    .section-desc {
        color: var(--text-muted);
        font-size: 13px;
        margin-top: -6px;
        margin-bottom: 12px;
    }

    .status-label {
        color: var(--text-muted);
        font-size: 13px;
        margin-bottom: 8px;
    }

    .status-value {
        color: var(--text-main);
        font-size: 22px;
        font-weight: 800;
    }

    .small-note {
        color: var(--text-muted);
        font-size: 12px;
        margin-top: 6px;
    }

    .upload-hint {
        color: var(--text-soft);
        font-size: 13px;
        line-height: 1.7;
        margin-top: 6px;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.72);
        border: 1.5px dashed rgba(96,165,250,0.35);
        border-radius: 18px;
        padding: 14px;
    }

    [data-testid="stFileUploader"] section {
        padding: 8px 4px;
    }

    .stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 14px;
    border: 1px solid rgba(59,130,246,0.18);
    background: linear-gradient(135deg, #2563eb, #60a5fa);
    color: white;
    font-weight: 700;
    font-size: 14px;
    transition: all 0.22s ease;
    box-shadow: 0 10px 22px rgba(37,99,235,0.16);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(37,99,235,0.22);
        border-color: rgba(59,130,246,0.28);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        margin-bottom: 8px;
    }

    .stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.70);
    border: 1px solid rgba(96,165,250,0.15);
    border-radius: 12px;
    padding: 8px 18px;
    color: var(--text-soft);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(219,234,254,0.95), rgba(191,219,254,0.95));
        color: #1d4ed8 !important;
        border: 1px solid rgba(59,130,246,0.22);
    }

    [data-testid="metric-container"] {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(96,165,250,0.16);
    padding: 16px 14px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(37,99,235,0.08);
    }

    .stDataFrame, div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    div[data-baseweb="notification"] {
        border-radius: 16px;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--primary-2));
    }

    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 14px 0;
    }
    header[data-testid="stHeader"] {
    display: none;
    }
    
    [data-testid="stToolbar"] {
        display: none;
    }
    
    [data-testid="stDecoration"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

def background_run_pipeline(file_bytes, file_name, result_queue):
    try:
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = file_name

        result = run_pipeline(
            file_obj=file_obj,
            control_state=None
        )

        result_queue.put({
            "status": "success",
            "result": result,
            "error": None
        })

    except Exception as e:
        result_queue.put({
            "status": "error",
            "result": None,
            "error": str(e)
        })

def render_header():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">📈 电商用户购买意向预测系统</div>
        <div class="hero-subtitle">
            面向电商行为分析与购买意向预测的机器学习平台。<br>
            支持数据上传、任务运行、过程追踪、指标评估与结果可视化展示。
        </div>
    </div>
    """, unsafe_allow_html=True)
CHART_COLORS = [
    "#2563eb",  # 蓝
    "#38bdf8",  # 天蓝
    "#0ea5e9",  # 湖蓝
    "#8b5cf6",  # 紫
    "#14b8a6",  # 青绿
    "#f59e0b",  # 橙
    "#ef4444",  # 红
    "#84cc16"   # 黄绿
]

PIE_COLORS = ["#2563eb", "#93c5fd", "#38bdf8", "#8b5cf6", "#14b8a6", "#f59e0b"]
def apply_chart_theme(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(255,255,255,0.78)",
        plot_bgcolor="rgba(239,246,255,0.85)",
        font=dict(color="#1e293b", size=13),
        title=dict(
            x=0.02,
            xanchor="left",
            font=dict(size=18, color="#0f172a")
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            bgcolor="rgba(255,255,255,0.55)",
            bordercolor="rgba(96,165,250,0.15)",
            borderwidth=1,
            font=dict(color="#334155")
        )
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(148,163,184,0.35)",
        tickfont=dict(color="#475569")
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False,
        tickfont=dict(color="#475569")
    )
    return fig
inject_custom_css()
render_header()

if "task_status" not in st.session_state:
    st.session_state.task_status = "idle"
if "pause_flag" not in st.session_state:
    st.session_state.pause_flag = False
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "result" not in st.session_state:
    st.session_state.result = None
if "worker_running" not in st.session_state:
    st.session_state.worker_running = False
if "worker_thread" not in st.session_state:
    st.session_state.worker_thread = None
if "run_error" not in st.session_state:
    st.session_state.run_error = None
if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

st.markdown('<div class="section-title">⚙️ 任务控制台</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">上传待分析文件，启动模型任务，并实时查看处理状态。</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1.15, 1])

with left_col:
    st.markdown("#### 📂 数据上传")

    uploaded_events = st.file_uploader(
        "上传 events 数据文件",
        type=["csv", "xlsx"],
        help="支持 CSV / XLSX 文件，用于模型预测与分析"
    )
    
    # ==========================
    # 优先使用上传文件
    # ==========================
    
    if uploaded_events is not None:
    
        st.session_state.uploaded_file_bytes = uploaded_events.getvalue()
        st.session_state.uploaded_file_name = uploaded_events.name
    
        st.success(f"已加载文件：{uploaded_events.name}")
        data_source = "upload"
    
    # ==========================
    # 否则使用默认数据
    # ==========================

else:

    if os.path.exists(DEFAULT_EVENTS_PATH):

        st.info("未上传文件，使用默认数据集")
        st.session_state.uploaded_file_bytes = None
        st.session_state.uploaded_file_name = DEFAULT_EVENTS_PATH
        data_source = "default"

    else:

        st.markdown(
            '<div class="upload-hint">支持 CSV / XLSX 格式。建议上传清洗后的事件数据，以获得更稳定的预测结果。</div>',
            unsafe_allow_html=True
        )
        data_source = None

with right_col:
    st.markdown("#### 🎛️ 运行控制")
    st.markdown('<div class="small-note">开始、暂停、继续或终止当前预测任务。</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    start_clicked = col1.button("▶ 开始运行", use_container_width=True)
    pause_clicked = col2.button("⏸ 暂停任务", use_container_width=True)
    resume_clicked = col3.button("⏯ 继续运行", use_container_width=True)
    stop_clicked = col4.button("⏹ 停止任务", use_container_width=True)


if start_clicked:
    if st.session_state.uploaded_file_bytes is None:
        st.warning("请先上传 events 文件，再开始运行。")
    elif st.session_state.worker_running:
        st.info("任务已在运行中，请勿重复启动。")
    else:
        st.session_state.task_status = "running"
        st.session_state.pause_flag = False
        st.session_state.stop_flag = False
        st.session_state.start_time = time.time()
        st.session_state.result = None
        st.session_state.run_error = None
        st.session_state.worker_running = True
        while not st.session_state.result_queue.empty():
            try:
                st.session_state.result_queue.get_nowait()
            except queue.Empty:
                break

        worker = threading.Thread(
            target=background_run_pipeline,
            args=(
                st.session_state.uploaded_file_bytes,
                st.session_state.uploaded_file_name,
                st.session_state.result_queue
            ),
            daemon=True
        )
        st.session_state.worker_thread = worker
        worker.start()


if pause_clicked:
    st.session_state.pause_flag = True
    st.session_state.task_status = "paused"

if resume_clicked:
    st.session_state.pause_flag = False
    st.session_state.task_status = "running"

if stop_clicked:
    st.session_state.stop_flag = True
    st.session_state.task_status = "stopped"


control_state = st.session_state


status_map = {
    "idle": "未开始",
    "running": "运行中",
    "paused": "已暂停",
    "stopped": "已停止",
    "done": "已完成"
}

display_status = status_map.get(st.session_state.task_status, st.session_state.task_status)

run_seconds = 0
if st.session_state.start_time:
    run_seconds = int(time.time() - st.session_state.start_time)

s1, s2 = st.columns(2)

with s1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="status-label">当前状态</div>
        <div class="status-value">{display_status}</div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="status-label">运行时间</div>
        <div class="status-value">{run_seconds} 秒</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.task_status == "running":
        try:
            worker_msg = st.session_state.result_queue.get_nowait()

            st.session_state.worker_running = False

            if worker_msg["status"] == "error":
                st.session_state.run_error = worker_msg["error"]
                st.session_state.task_status = "stopped"
            else:
                st.session_state.result = worker_msg["result"]
                st.session_state.task_status = "done"

            st.rerun()

        except queue.Empty:
            time.sleep(1)
            st.rerun()

if st.session_state.result is not None:
    st.success("运行完成")

    tab1, tab2 = st.tabs(["结果可视化", "模型与数据概览"])

    with tab1:
        st.subheader("结果可视化")
        metrics = st.session_state.result.get("visual_metrics", {})

        card1, card2, card3, card4 = st.columns(4)
        card1.metric("准确率 Accuracy",
                     f"{metrics.get('accuracy', 0):.4f}" if metrics.get("accuracy") is not None else "N/A")
        card2.metric("精准率 Precision",
                     f"{metrics.get('precision', 0):.4f}" if metrics.get("precision") is not None else "N/A")
        card3.metric("召回率 Recall", f"{metrics.get('recall', 0):.4f}" if metrics.get("recall") is not None else "N/A")
        card4.metric("F1", f"{metrics.get('f1', 0):.4f}" if metrics.get("f1") is not None else "N/A")

        card5, card6, card7, card8 = st.columns(4)
        card5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}" if metrics.get("roc_auc") is not None else "N/A")
        card6.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}" if metrics.get("pr_auc") is not None else "N/A")
        card7.metric("正确率",
                     f"{metrics.get('correct_rate', 0):.4f}" if metrics.get("correct_rate") is not None else "N/A")
        card8.metric("错误率",
                     f"{metrics.get('error_rate', 0):.4f}" if metrics.get("error_rate") is not None else "N/A")

        left_chart, right_chart = st.columns(2)

        metric_df = pd.DataFrame({
            "指标": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
            "数值": [
                metrics.get("accuracy", 0),
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1", 0),
                metrics.get("roc_auc", 0),
                metrics.get("pr_auc", 0),
            ]
        })

        fig_bar = px.bar(
            metric_df,
            x="指标",
            y="数值",
            color="指标",
            color_discrete_sequence=CHART_COLORS,
            title="模型核心评估指标"
        )
        fig_bar.update_traces(
            texttemplate="%{y:.3f}",
            textposition="outside",
            marker_line_width=0
        )
        fig_bar = apply_chart_theme(fig_bar, height=360)
        fig_bar.update_layout(showlegend=False)
        fig_bar = apply_chart_theme(fig_bar, height=360)

        pie_df = pd.DataFrame({
            "类别": ["正确", "错误"],
            "数值": [
                metrics.get("correct_rate", 0),
                metrics.get("error_rate", 0)
            ]
        })

        fig_pie = px.pie(
            pie_df,
            names="类别",
            values="数值",
            title="正确率 / 错误率分布",
            hole=0.5,
            color="类别",
            color_discrete_map={
                "正确": "#2563eb",
                "错误": "#f59e0b"
            }
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            pull=[0.02, 0.02]
        )
        fig_pie = apply_chart_theme(fig_pie, height=360)

        with left_chart:
            st.plotly_chart(fig_bar, use_container_width=True)

        with right_chart:
            st.plotly_chart(fig_pie, use_container_width=True)

        left_chart2, right_chart2 = st.columns(2)

        conf_df = pd.DataFrame({
            "类型": ["TP", "TN", "FP", "FN"],
            "数量": [
                metrics.get("tp", 0),
                metrics.get("tn", 0),
                metrics.get("fp", 0),
                metrics.get("fn", 0),
            ]
        })

        fig_conf = px.bar(
            conf_df,
            x="类型",
            y="数量",
            color="类型",
            color_discrete_map={
                "TP": "#2563eb",
                "TN": "#14b8a6",
                "FP": "#f59e0b",
                "FN": "#ef4444"
            },
            title="混淆矩阵构成"
        )
        fig_conf.update_traces(
            texttemplate="%{y}",
            textposition="outside"
        )
        fig_conf = apply_chart_theme(fig_conf, height=360)
        fig_conf.update_layout(showlegend=False)
        fig_conf = apply_chart_theme(fig_conf, height=360)

        with left_chart2:
            st.plotly_chart(fig_conf, use_container_width=True)

        with right_chart2:
            if "importance" in st.session_state.result:
                imp_df = st.session_state.result["importance"].head(15).copy()
                fig_imp = px.bar(
                    imp_df,
                    x="importance_mean",
                    y="feature",
                    orientation="h",
                    color="importance_mean",
                    color_continuous_scale=[
                        [0.0, "#dbeafe"],
                        [0.25, "#93c5fd"],
                        [0.5, "#60a5fa"],
                        [0.75, "#3b82f6"],
                        [1.0, "#1d4ed8"]
                    ],
                    title="Top 15 特征重要性"
                )
                fig_imp.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False
                )
                fig_imp.update_traces(marker_line_width=0)
                fig_imp = apply_chart_theme(fig_imp, height=360)
                fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
                fig_imp = apply_chart_theme(fig_imp, height=360)
                st.plotly_chart(fig_imp, use_container_width=True)

        if "dataset" in st.session_state.result:
            ds = st.session_state.result["dataset"].copy()
            if "label_purchase_next_window" in ds.columns:
                label_df = (
                    ds["label_purchase_next_window"]
                    .value_counts()
                    .rename_axis("label")
                    .reset_index(name="count")
                )
                fig_label = px.pie(
                    label_df,
                    names="label",
                    values="count",
                    title="监督数据集标签分布",
                    hole=0.45,
                    color="label",
                    color_discrete_sequence=PIE_COLORS
                )
                fig_label.update_traces(textinfo="percent+label")
                fig_label = apply_chart_theme(fig_label, height=380)
                st.plotly_chart(fig_label, use_container_width=True)

    with tab2:
        if "cv_metrics" in st.session_state.result and st.session_state.result["cv_metrics"]:
            st.markdown("#### 📈 Rolling CV 结果")
            cv_df = pd.DataFrame(st.session_state.result["cv_metrics"])
            st.dataframe(cv_df, use_container_width=True)

            if "pr_auc" in cv_df.columns:
                fig_cv = px.line(
                    cv_df,
                    x="fold",
                    y="pr_auc",
                    markers=True,
                    title="Rolling CV - PR-AUC 变化趋势"
                )
                fig_cv.update_traces(
                    line=dict(color="#2563eb", width=3),
                    marker=dict(size=9, color="#38bdf8", line=dict(width=2, color="#ffffff"))
                )
                fig_cv = apply_chart_theme(fig_cv, height=360)
                st.plotly_chart(fig_cv, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### 📋 监督数据集预览")
        st.dataframe(
            st.session_state.result["dataset"].head(50),
            use_container_width=True
        )

        if "importance" in st.session_state.result:
            st.markdown("#### 🔍 特征重要性 Top 30")
            st.dataframe(
                st.session_state.result["importance"].head(30),
                use_container_width=True
            )

        st.markdown("#### 🧠 最佳模型信息")
        st.json(st.session_state.result["best_meta"])
        st.markdown('</div>', unsafe_allow_html=True)
