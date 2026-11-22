"""Streamlit dashboard for the completed energy and FL experiments."""

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
    page_title="IoT Federated Learning Results",
    page_icon="📊",
    layout="wide",
)

st.title("Security-Aware and Energy-Efficient FL for IoT Intrusion Detection")
st.caption(
    "BENIGN vs DDoS • Five simulated IoT clients • "
    "Energy values are estimates, not hardware measurements"
)

results_path = RESULTS_DIR / "experiment_results.csv"
participation_path = WEEK3_RESULTS_DIR / "participation.csv"
round_metrics_path = WEEK3_RESULTS_DIR / "round_metrics.csv"
required_paths = [results_path, participation_path, round_metrics_path]
missing_paths = [path for path in required_paths if not path.exists()]
if missing_paths:
    st.error(
        "Required experiment results are missing. Run compare_results.py "
        "and the Week 3 comparison first."
    )
    for path in missing_paths:
        st.code(str(path))
    st.stop()

results = pd.read_csv(results_path)
participation = pd.read_csv(participation_path)
round_metrics = pd.read_csv(round_metrics_path)

st.subheader("Experiment results")
st.dataframe(results, hide_index=True, width="stretch")

unavailable = results[results["Status"] == "unavailable"]
if not unavailable.empty:
    st.warning(
        "Security-defended federated experiments are unavailable because "
        "the repository did not include a federated security-defense module. "
        "Their metrics are intentionally blank."
    )

standard_rows = results[results["Experiment"] == "Standard FedAvg"]
aware_rows = results[results["Experiment"] == "Energy-aware FedAvg"]
if not standard_rows.empty and not aware_rows.empty:
    standard = standard_rows.iloc[0]
    aware = aware_rows.iloc[0]
    columns = st.columns(4)
    columns[0].metric(
        "Energy-aware accuracy",
        f"{float(aware['Accuracy']) * 100:.3f}%",
    )
    columns[1].metric(
        "DDoS recall",
        f"{float(aware['Recall']) * 100:.3f}%",
    )
    columns[2].metric(
        "False-positive rate",
        f"{float(aware['False_Positive_Rate']) * 100:.3f}%",
    )
    columns[3].metric(
        "Energy-aware simulated energy",
        f"{float(aware['Total_Simulated_Energy_Used']):.2f} units",
    )

    standard_energy = float(standard["Total_Simulated_Energy_Used"])
    aware_energy = float(aware["Total_Simulated_Energy_Used"])
    if standard_energy > 0:
        savings = (standard_energy - aware_energy) / standard_energy * 100
        st.metric(
            "Estimated simulated energy reduction vs Standard FedAvg",
            f"{savings:.2f}%",
            help="A simulation estimate based on configured costs, not battery data.",
        )

st.subheader("Client participation by round")
participation["Participated"] = (
    participation["Participated"]
    .astype(str)
    .str.lower()
    .map({"true": True, "false": False})
)
if participation["Participated"].isna().any():
    raise ValueError("Participation CSV contains invalid status values.")

participation_counts = (
    participation.groupby(["Round", "Strategy"])["Participated"]
    .sum()
    .unstack("Strategy")
)
st.dataframe(participation_counts, width="stretch")

st.subheader("Simulated energy consumption")
energy_by_round = round_metrics.pivot(
    index="Round",
    columns="Strategy",
    values="Energy_Used",
)
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


@st.cache_resource
def run_request_security_demo(test_path):
    """Run one supplied-record security demonstration per cached process."""
    test_data = pd.read_csv(test_path)

    if "Label" not in test_data.columns:
        raise ValueError("Independent test data has no Label column.")

    normal_rows = test_data[test_data["Label"] == 0]
    ddos_rows = test_data[test_data["Label"] == 1]
    if normal_rows.empty or ddos_rows.empty:
        raise ValueError("Independent test data needs normal and DDoS records.")

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

    return normal_result, ddos_result, repeated_results, invalid_result, counts


st.subheader("IoT Request Security")
st.caption(
    "Supplied IoT feature-record demonstration using the existing "
    "classifier and 60-second rate limiter."
)

if not TEST_FILE.exists():
    st.warning(
        "IoT request demonstration unavailable: "
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
        st.warning(
            "IoT request demonstration unavailable: "
            f"{error}"
        )
    else:
        first_demo, ddos_demo, repeated_demo = st.columns(3)
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
        repeated_demo.metric(
            "Repeated normal requests",
            "Rate limited" if repeated_limited else "Allowed",
            f"{allowed_repeated} allowed, {repeated_limited} limited",
        )

        count_columns = st.columns(4)
        count_columns[0].metric("Allowed", request_counts["allowed"])
        count_columns[1].metric("Blocked", request_counts["blocked"])
        count_columns[2].metric("Limited", request_counts["limited"])
        count_columns[3].metric("Invalid", request_counts["invalid"])

        with st.expander("Request reasons"):
            st.write(normal_result["reason"])
            st.write(ddos_result["reason"])
            st.write(repeated_results[-1]["reason"])
            st.write(invalid_result["reason"])


st.subheader("Presentation graphs")
graph_files = [
    GRAPH_DIR / "energy_per_client.png",
    GRAPH_DIR / "energy_per_round.png",
    GRAPH_DIR / "participants_per_round.png",
    GRAPH_DIR / "accuracy_comparison.png",
    GRAPH_DIR / "energy_savings_comparison.png",
]
for graph_path in graph_files:
    if graph_path.exists():
        st.image(str(graph_path), caption=graph_path.stem.replace("_", " ").title())
    else:
        st.info(
            f"Graph not generated yet: {graph_path.name}. "
            "Run visualize_results.py."
        )
