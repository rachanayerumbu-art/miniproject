"""Streamlit dashboard for saved federated-learning research results."""

from pathlib import Path

import pandas as pd
import streamlit as st

from security.request_handler import handle_request
from security.rate_limiter import reset_rate_limiter


PROJECT_DIR = Path(__file__).resolve().parent
WEEK3_RESULTS_DIR = PROJECT_DIR / "week3_energy_comparison"
RESULTS_DIR = PROJECT_DIR / "week4_results"
GRAPH_DIR = RESULTS_DIR / "presentation_graphs"
TEST_FILE = PROJECT_DIR / "data" / "independent_test.csv"

st.set_page_config(
    page_title="IoT Federated Learning Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --canvas: #080d17;
        --panel: #0e1624;
        --panel-raised: #121d2d;
        --line: rgba(130, 165, 190, 0.16);
        --text: #e8f0f7;
        --muted: #91a2b5;
        --cyan: #45d9e8;
        --teal: #39d6b1;
        --violet: #9c8cff;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(52, 124, 160, 0.13), transparent 34%),
            radial-gradient(ellipse at 0% 40%, rgba(98, 79, 159, 0.09), transparent 30%),
            var(--canvas);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 23, 0.75);
    }

    [data-testid="stSidebar"] {
        background: #0b111c;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #91a2b5 !important;
        opacity: 1;
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.025em;
    }

    h1 {
        font-size: clamp(1.8rem, 3vw, 2.45rem) !important;
        font-weight: 650 !important;
    }

    h2 {
        font-size: 1.45rem !important;
        font-weight: 620 !important;
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(18, 29, 45, 0.96), rgba(12, 21, 34, 0.9));
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 112px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--text);
        font-size: clamp(1.25rem, 2vw, 1.75rem) !important;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-baseweb="tab-list"] {
        gap: 8px;
    }

    .research-hero {
        border: 1px solid rgba(69, 217, 232, 0.24);
        border-radius: 20px;
        padding: clamp(22px, 4vw, 38px);
        margin: 0 0 24px;
        background:
            radial-gradient(ellipse at 82% 0%, rgba(69, 217, 232, 0.13), transparent 42%),
            linear-gradient(125deg, rgba(18, 31, 48, 0.97), rgba(12, 19, 32, 0.97));
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.16), 0 0 28px rgba(69, 217, 232, 0.035);
    }

    .research-kicker {
        color: var(--cyan);
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .research-title {
        color: var(--text);
        font-size: clamp(1.8rem, 3.2vw, 2.65rem);
        font-weight: 660;
        line-height: 1.13;
        letter-spacing: -0.04em;
        max-width: 900px;
    }

    .research-description {
        color: #a7b8c8;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 850px;
        margin: 14px 0 0;
    }

    .section-heading {
        color: var(--text);
        font-size: 1.45rem;
        font-weight: 620;
        letter-spacing: -0.025em;
        margin: 4px 0 5px;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        margin: 0 0 20px;
    }

    .status-strip {
        background: rgba(13, 23, 37, 0.86);
        border: 1px solid var(--line);
        border-radius: 12px;
        color: #a8b8c8;
        font-size: 0.88rem;
        line-height: 1.6;
        padding: 13px 16px;
        margin: 0 0 18px;
    }

    .accent-cyan { color: var(--cyan); }
    .accent-teal { color: var(--teal); }
    .accent-violet { color: var(--violet); }

    @media (max-width: 720px) {
        [data-testid="stMetric"] {
            padding: 12px;
            min-height: 96px;
        }
        .research-hero {
            border-radius: 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_saved_csv(path):
    if not path.exists():
        return None, f"Saved file not found: {path.relative_to(PROJECT_DIR)}"

    try:
        return pd.read_csv(path), None
    except (OSError, ValueError, pd.errors.ParserError) as error:
        return None, f"Could not read {path.relative_to(PROJECT_DIR)}: {error}"


def metric_value(row, column, percent=False):
    if row is None or column not in row or pd.isna(row[column]):
        return "—"
    value = float(row[column])
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def experiment_row(frame, experiment):
    if frame is None or "Experiment" not in frame.columns:
        return None
    rows = frame.loc[frame["Experiment"] == experiment]
    return None if rows.empty else rows.iloc[0]


def show_saved_graph(filename, caption):
    graph_path = GRAPH_DIR / filename
    if graph_path.exists():
        st.image(str(graph_path), caption=caption)
    else:
        st.info(
            f"Saved graph not generated yet: {filename}. "
            "Run visualize_results.py."
        )


@st.cache_resource
def run_request_security_demo(test_path):
    """Run one supplied-record security demonstration per cached process."""
    test_data = pd.read_csv(test_path)

    if "Label" not in test_data.columns:
        raise ValueError("Independent test data has no Label column.")

    normal_rows = test_data[test_data["Label"] == 0]
    ddos_rows = test_data[test_data["Label"] == 1]
    if normal_rows.empty or ddos_rows.empty:
        raise ValueError(
            "Independent test data needs normal and DDoS records."
        )

    normal_record = normal_rows.iloc[0].drop(labels=["Label"]).to_dict()
    ddos_record = ddos_rows.iloc[0].drop(labels=["Label"]).to_dict()
    invalid_record = normal_record.copy()
    invalid_record.pop(next(iter(invalid_record)))

    reset_rate_limiter()
    fixed_time = lambda: 1000.0

    normal_result = handle_request(
        normal_record,
        "dashboard-normal-device",
        time_source=fixed_time,
    )
    ddos_result = handle_request(
        ddos_record,
        "dashboard-ddos-device",
        time_source=fixed_time,
    )
    repeated_results = [
        handle_request(
            normal_record,
            "dashboard-repeated-device",
            time_source=fixed_time,
        )
        for _ in range(11)
    ]
    invalid_result = handle_request(
        invalid_record,
        "dashboard-invalid-device",
        time_source=fixed_time,
    )

    results = [
        normal_result,
        ddos_result,
        *repeated_results,
        invalid_result,
    ]
    counts = {
        status: sum(
            result["status"] == status
            for result in results
        )
        for status in ("allowed", "blocked", "limited", "invalid")
    }

    return (
        normal_result,
        ddos_result,
        repeated_results,
        invalid_result,
        counts,
    )


results_path = RESULTS_DIR / "experiment_results.csv"
participation_path = WEEK3_RESULTS_DIR / "participation.csv"
round_metrics_path = WEEK3_RESULTS_DIR / "round_metrics.csv"

results, results_error = load_saved_csv(results_path)
participation, participation_error = load_saved_csv(participation_path)
round_metrics, round_metrics_error = load_saved_csv(round_metrics_path)

participation_ready = participation is not None
if participation_ready:
    participation = participation.copy()
    participation["Participated"] = (
        participation["Participated"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    if participation["Participated"].isna().any():
        participation_ready = False
        participation_error = (
            "Participation CSV contains invalid status values."
        )

results_available = results is not None
energy_data_available = (
    participation_ready and round_metrics is not None
)
standard = experiment_row(results, "Standard FedAvg")
energy_aware = experiment_row(results, "Energy-aware FedAvg")

st.sidebar.markdown(
    """
    <div style="padding: 10px 0 16px;">
      <div class="research-kicker">Research console</div>
      <div style="color:#e8f0f7;font-size:1.05rem;font-weight:650;">
        IoT / Federated AI
      </div>
      <div style="color:#8495a8;font-size:0.78rem;margin-top:5px;">
        Security and energy evaluation
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

section = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Federated Learning",
        "IoT Client Network",
        "Threat Detection",
        "Performance Analysis",
        "Energy Analysis",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("DATA STATUS")
if results_available and energy_data_available:
    st.sidebar.markdown("● Saved experiment files loaded")
else:
    st.sidebar.markdown("○ Some saved experiment files unavailable")
st.sidebar.caption("File-backed research dashboard · no live monitoring")

if section == "Overview":
    st.markdown(
        """
        <div class="research-hero">
          <div class="research-kicker">Security-aware · energy-efficient · federated</div>
          <div class="research-title">
            IoT Intrusion Detection with Federated Learning
          </div>
          <p class="research-description">
            Research dashboard for saved Standard FedAvg and energy-aware
            experiment results, simulated client participation, and a
            supplied-record request-security demonstration.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Research overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Historical experiment artifacts currently available in this workspace.</div>',
        unsafe_allow_html=True,
    )

    overview_metrics_top = st.columns(2)
    overview_metrics_bottom = st.columns(2)
    overview_metrics_top[0].metric(
        "Standard FedAvg accuracy",
        metric_value(standard, "Accuracy", percent=True),
    )
    overview_metrics_top[1].metric(
        "Energy-aware accuracy",
        metric_value(energy_aware, "Accuracy", percent=True),
    )
    client_count = (
        participation["Client"].nunique()
        if participation_ready and "Client" in participation.columns
        else "—"
    )
    overview_metrics_bottom[0].metric("Simulated clients", client_count)
    round_count = (
        int(participation["Round"].max())
        if participation_ready and "Round" in participation.columns
        else "—"
    )
    overview_metrics_bottom[1].metric("Recorded rounds", round_count)

    st.markdown(
        '<div class="status-strip"><span class="accent-cyan">SYSTEM STATUS</span> &nbsp; '
        "Offline research dashboard · displays saved CSV/PNG artifacts only. "
        "Energy values are simulation estimates, not device measurements. "
        "No training runs are started from this interface.</div>",
        unsafe_allow_html=True,
    )

    if results_available:
        st.subheader("Saved experiment results")
        st.dataframe(results, hide_index=True, width="stretch")
        unavailable = (
            results[results["Status"] == "unavailable"]
            if "Status" in results.columns
            else pd.DataFrame()
        )
        if not unavailable.empty:
            st.warning(
                "Some security-defense experiments are marked unavailable "
                "in the saved results. Their metrics remain blank."
            )
    else:
        st.warning(results_error)

elif section == "Federated Learning":
    st.markdown('<div class="section-heading">Federated learning experiments</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Saved Standard FedAvg and energy-aware results. '
        "Training controls are not enabled in this dashboard.</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "The figures below are loaded from existing historical result files; "
        "they are not newly executed experiments."
    )

    if results_available:
        metric_columns_top = st.columns(2)
        metric_columns_bottom = st.columns(2)
        metric_columns_top[0].metric(
            "Standard accuracy",
            metric_value(standard, "Accuracy", percent=True),
        )
        metric_columns_top[1].metric(
            "Energy-aware accuracy",
            metric_value(energy_aware, "Accuracy", percent=True),
        )
        metric_columns_bottom[0].metric(
            "Standard F1-score",
            metric_value(standard, "F1_Score"),
        )
        metric_columns_bottom[1].metric(
            "Energy-aware F1-score",
            metric_value(energy_aware, "F1_Score"),
        )
        st.dataframe(results, hide_index=True, width="stretch")
        if "Status" in results.columns:
            unavailable = results[results["Status"] == "unavailable"]
            if not unavailable.empty:
                st.warning(
                    "Security-defense experiment rows are unavailable in "
                    "the saved results and are not represented as metrics."
                )
    else:
        st.warning(results_error)

elif section == "IoT Client Network":
    st.markdown('<div class="section-heading">IoT client network</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Five simulated federated clients and their recorded participation.</div>',
        unsafe_allow_html=True,
    )

    if not participation_ready:
        st.warning(participation_error)
    else:
        if {"Round", "Strategy", "Participated"}.issubset(participation.columns):
            participation_counts = (
                participation.groupby(["Round", "Strategy"])["Participated"]
                .sum()
                .unstack("Strategy")
            )
            st.subheader("Participating clients by round")
            st.bar_chart(participation_counts)

        if {"Round", "Client", "Strategy", "Participated"}.issubset(
            participation.columns
        ):
            latest_round = participation["Round"].max()
            latest = participation.loc[
                participation["Round"] == latest_round,
                ["Client", "Strategy", "Participated"],
            ]
            latest_network = latest.pivot(
                index="Client",
                columns="Strategy",
                values="Participated",
            )
            st.subheader(f"Latest recorded round · {latest_round}")
            st.dataframe(latest_network, width="stretch")

        st.subheader("Participation history")
        st.dataframe(participation, hide_index=True, width="stretch")

elif section == "Threat Detection":
    st.markdown('<div class="section-heading">Threat detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Deterministic demonstration using supplied records from the independent test dataset.</div>',
        unsafe_allow_html=True,
    )

    if not TEST_FILE.exists():
        st.warning(
            "Request demonstration unavailable: "
            "data/independent_test.csv was not found."
        )
    else:
        try:
            (
                normal_result,
                ddos_result,
                repeated_results,
                invalid_result,
                request_counts,
            ) = run_request_security_demo(str(TEST_FILE))
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.warning(f"Request demonstration unavailable: {error}")
        else:
            st.caption(
                "Request-handling demonstration only; this is not live "
                "network monitoring or firewall protection."
            )
            first_demo, ddos_demo = st.columns(2)
            first_demo.metric(
                "Normal request",
                normal_result["status"].title(),
            )
            ddos_demo.metric(
                "DDoS request",
                ddos_result["status"].title(),
            )

            allowed_repeated = sum(
                result["status"] == "allowed"
                for result in repeated_results
            )
            repeated_limited = sum(
                result["status"] == "limited"
                for result in repeated_results
            )
            st.metric(
                "Repeated normal requests",
                "Rate limited" if repeated_limited else "Allowed",
            )
            st.caption(
                f"{allowed_repeated} allowed, {repeated_limited} limited "
                "across 11 supplied-record requests."
            )

            count_columns = st.columns(4)
            count_columns[0].metric(
                "Allowed",
                request_counts["allowed"],
            )
            count_columns[1].metric(
                "Blocked",
                request_counts["blocked"],
            )
            count_columns[2].metric(
                "Limited",
                request_counts["limited"],
            )
            count_columns[3].metric(
                "Invalid",
                request_counts["invalid"],
            )

            with st.expander("Request reasons"):
                st.write(normal_result["reason"])
                st.write(ddos_result["reason"])
                st.write(repeated_results[-1]["reason"])
                st.write(invalid_result["reason"])

elif section == "Performance Analysis":
    st.markdown('<div class="section-heading">Performance analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Saved independent-test metrics; unavailable values are not estimated.</div>',
        unsafe_allow_html=True,
    )

    if results_available:
        performance_columns_top = st.columns(2)
        performance_columns_bottom = st.columns(2)
        performance_columns_top[0].metric(
            "Accuracy · energy-aware",
            metric_value(energy_aware, "Accuracy", percent=True),
        )
        performance_columns_top[1].metric(
            "Precision · energy-aware",
            metric_value(energy_aware, "Precision"),
        )
        performance_columns_bottom[0].metric(
            "Recall · energy-aware",
            metric_value(energy_aware, "Recall"),
        )
        performance_columns_bottom[1].metric(
            "F1-score · energy-aware",
            metric_value(energy_aware, "F1_Score"),
        )

        metric_fields = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
        ]
        if "Experiment" in results.columns and all(
            field in results.columns for field in metric_fields
        ):
            performance_data = results[
                results["Status"].eq("completed")
                if "Status" in results.columns
                else results["Experiment"].notna()
            ][["Experiment"] + metric_fields].copy()
            for field in metric_fields:
                performance_data[field] = pd.to_numeric(
                    performance_data[field],
                    errors="coerce",
                )
            performance_data = performance_data.dropna(
                subset=metric_fields,
                how="all",
            )
            if not performance_data.empty:
                st.subheader("Saved metric comparison")
                st.bar_chart(performance_data.set_index("Experiment"))

        st.subheader("Saved result table")
        st.dataframe(results, hide_index=True, width="stretch")
    else:
        st.warning(results_error)

    st.subheader("Confusion matrix")
    st.info(
        "A confusion matrix is not stored in the dashboard's saved result "
        "CSV files. The trainer prints one to its console, but no matrix is "
        "available here to display without a newly saved result."
    )
    show_saved_graph(
        "accuracy_comparison.png",
        "Saved historical accuracy comparison",
    )

elif section == "Energy Analysis":
    st.markdown('<div class="section-heading">Energy analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Participation and estimated energy use from saved simulations.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Energy values are illustrative simulation units, not measurements "
        "from physical devices."
    )

    if not energy_data_available:
        st.warning(
            participation_error
            if not participation_ready
            else round_metrics_error
        )
    else:
        if standard is not None and energy_aware is not None:
            energy_columns = st.columns(3)
            energy_columns[0].metric(
                "Standard simulated energy",
                metric_value(standard, "Total_Simulated_Energy_Used"),
            )
            energy_columns[1].metric(
                "Energy-aware simulated energy",
                metric_value(energy_aware, "Total_Simulated_Energy_Used"),
            )
            standard_energy = standard.get(
                "Total_Simulated_Energy_Used"
            )
            aware_energy = energy_aware.get(
                "Total_Simulated_Energy_Used"
            )
            if (
                pd.notna(standard_energy)
                and pd.notna(aware_energy)
                and float(standard_energy) > 0
            ):
                savings = (
                    float(standard_energy) - float(aware_energy)
                ) / float(standard_energy) * 100
                energy_columns[2].metric(
                    "Estimated simulated reduction",
                    f"{savings:.2f}%",
                    help=(
                        "A simulation estimate based on configured costs, "
                        "not physical energy measurements."
                    ),
                )
            else:
                energy_columns[2].metric(
                    "Estimated simulated reduction",
                    "—",
                )

        participation_counts = (
            participation.groupby(["Round", "Strategy"])["Participated"]
            .sum()
            .unstack("Strategy")
        )
        st.subheader("Client participation by round")
        st.bar_chart(participation_counts)
        st.dataframe(participation_counts, width="stretch")

        energy_by_round = round_metrics.pivot(
            index="Round",
            columns="Strategy",
            values="Energy_Used",
        )
        st.subheader("Simulated energy consumption")
        st.dataframe(energy_by_round, width="stretch")

        left, right = st.columns(2)
        with left:
            st.subheader("Energy used per client")
            energy_by_client = (
                participation.groupby(["Client", "Strategy"])["Energy_Used"]
                .sum()
                .unstack("Strategy")
                .fillna(0)
            )
            st.dataframe(energy_by_client, width="stretch")

        with right:
            st.subheader("Remaining client energy after the last round")
            final_energy = (
                participation.sort_values("Round")
                .groupby(["Strategy", "Client"], as_index=False)
                .tail(1)
                .pivot(
                    index="Client",
                    columns="Strategy",
                    values="Energy_Remaining",
                )
            )
            st.dataframe(final_energy, width="stretch")

        st.subheader("Saved energy and participation graphs")
        show_saved_graph(
            "energy_per_client.png",
            "Saved historical energy used per client",
        )
        show_saved_graph(
            "energy_per_round.png",
            "Saved historical energy use per round",
        )
        show_saved_graph(
            "participants_per_round.png",
            "Saved historical client participation per round",
        )
        show_saved_graph(
            "energy_savings_comparison.png",
            "Saved historical estimated energy savings",
        )
