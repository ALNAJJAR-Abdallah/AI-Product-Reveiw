"""
Streamlit application for AI Product Review Intelligence System.

Professional UI version:
- Modern dashboard design
- Product analysis workflow
- Sentiment KPIs
- Styled cards
- Human-in-the-loop validation
- Final report generation
- JSON logs viewer
"""

import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv

from src.orchestration.orchestrator import ProductReviewOrchestrator
from src.utils.errors import HumanApprovalRequiredError, ProductReviewError
from src.utils.logger import read_logs


# =========================
# ENVIRONMENT CONFIGURATION
# =========================

load_dotenv()


# =========================
# STREAMLIT PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Product Review Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# CUSTOM CSS
# =========================

st.markdown(
    """
    <style>
        /* ==============================
           GLOBAL THEME
        ============================== */
        .stApp {
            background: linear-gradient(135deg, #000000 0%, #0a0a0a 35%, #111111 70%, #1a1a1a 100%);
            color: #f5f5f5;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ==============================
           SIDEBAR
        ============================== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #050505 0%, #111111 100%);
            border-right: 1px solid rgba(212, 175, 55, 0.18);
        }

        section[data-testid="stSidebar"] * {
            color: #f3f3f3;
        }

        /* ==============================
           HERO SECTION
        ============================== */
        .hero-card {
            padding: 2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(20, 20, 20, 0.95), rgba(40, 40, 40, 0.85));
            border: 1px solid rgba(212, 175, 55, 0.28);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.5rem;
            color: #f8f8f8;
            letter-spacing: 0.3px;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #d6d6d6;
            max-width: 900px;
        }

        .hero-badge {
            display: inline-block;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            background: rgba(212, 175, 55, 0.10);
            border: 1px solid rgba(212, 175, 55, 0.35);
            color: #f5d27a;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        /* ==============================
           GLASS / PREMIUM CARDS
        ============================== */
        .glass-card {
            padding: 1.2rem;
            border-radius: 20px;
            background: rgba(18, 18, 18, 0.88);
            border: 1px solid rgba(212, 175, 55, 0.16);
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
            margin-bottom: 1rem;
        }

        .small-card-title {
            font-size: 0.9rem;
            color: #c8b273;
            margin-bottom: 0.3rem;
            font-weight: 600;
        }

        .small-card-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #ffffff;
        }

        .small-card-caption {
            font-size: 0.8rem;
            color: #b8b8b8;
            margin-top: 0.2rem;
        }

        /* ==============================
           TITLES
        ============================== */
        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #f5d27a;
            margin-top: 1rem;
            margin-bottom: 0.6rem;
        }

        .section-subtitle {
            font-size: 0.9rem;
            color: #bdbdbd;
            margin-bottom: 1rem;
        }

        /* ==============================
           PILLS / TAGS
        ============================== */
        .pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(30, 30, 30, 0.95);
            border: 1px solid rgba(212, 175, 55, 0.16);
            color: #f1f1f1;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }

        .positive-pill {
            background: rgba(40, 70, 40, 0.35);
            border: 1px solid rgba(97, 173, 97, 0.35);
            color: #cceccc;
        }

        .negative-pill {
            background: rgba(90, 25, 25, 0.35);
            border: 1px solid rgba(210, 80, 80, 0.35);
            color: #f7caca;
        }

        /* ==============================
           BUTTONS
        ============================== */
        div.stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(212, 175, 55, 0.35);
            background: linear-gradient(135deg, #b8860b, #d4af37, #f5d27a);
            color: #111111;
            font-weight: 800;
            padding: 0.65rem 1rem;
            transition: 0.2s ease-in-out;
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 30px rgba(212, 175, 55, 0.22);
            border: 1px solid rgba(245, 210, 122, 0.65);
        }

        /* ==============================
           INPUTS
        ============================== */
        .stTextInput input, .stTextArea textarea {
            background: rgba(14, 14, 14, 0.95) !important;
            color: #f8f8f8 !important;
            border-radius: 14px !important;
            border: 1px solid rgba(212, 175, 55, 0.18) !important;
        }

        .stTextInput input:focus, .stTextArea textarea:focus {
            border: 1px solid rgba(212, 175, 55, 0.75) !important;
            box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.12) !important;
        }

        /* ==============================
           RADIO / CHECKBOX / SLIDER TEXT
        ============================== */
        .stRadio label, .stCheckbox label, .stSlider label {
            color: #f1f1f1 !important;
        }

        /* ==============================
           DATAFRAME
        ============================== */
        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(212, 175, 55, 0.16);
        }

        /* ==============================
           EXPANDER
        ============================== */
        .streamlit-expanderHeader {
            background: rgba(20, 20, 20, 0.9);
            border-radius: 12px;
            color: #f5d27a;
        }

        /* ==============================
           METRICS
        ============================== */
        div[data-testid="stMetric"] {
            background: rgba(18, 18, 18, 0.88);
            border: 1px solid rgba(212, 175, 55, 0.14);
            padding: 1rem;
            border-radius: 18px;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff;
            font-weight: 800;
        }

        div[data-testid="stMetricLabel"] {
            color: #d4c08a;
        }

        /* ==============================
           INFO / SUCCESS / WARNING / ERROR
        ============================== */
        div[data-testid="stAlert"] {
            border-radius: 14px;
        }

        /* ==============================
           TABS
        ============================== */
        button[data-baseweb="tab"] {
            background-color: rgba(18, 18, 18, 0.85);
            color: #d8d8d8;
            border-radius: 10px 10px 0 0;
            border: 1px solid rgba(212, 175, 55, 0.10);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #f5d27a;
            border-bottom: 2px solid #d4af37;
        }

        /* ==============================
           HORIZONTAL RULE
        ============================== */
        hr {
            border: none;
            border-top: 1px solid rgba(212, 175, 55, 0.12);
            margin: 1.5rem 0;
        }

        /* ==============================
        PREMIUM TABS CONTAINER
        ============================== */
        div[data-testid="stTabs"] {
            background: rgba(10, 10, 10, 0.92);
            border: 1px solid rgba(212, 175, 55, 0.18);
            border-radius: 20px;
            padding: 1rem;
            margin-top: 1rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
        }

        /* Tabs row */
        div[data-testid="stTabs"] div[role="tablist"] {
            gap: 0.5rem;
            background: rgba(18, 18, 18, 0.95);
            border: 1px solid rgba(212, 175, 55, 0.12);
            border-radius: 16px;
            padding: 0.45rem;
            margin-bottom: 1.4rem;
        }

        /* Individual tabs */
        button[data-baseweb="tab"] {
            background: transparent !important;
            color: #d8d8d8 !important;
            border-radius: 12px !important;
            border: none !important;
            padding: 0.65rem 1rem !important;
        }

        /* Active tab */
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(212, 175, 55, 0.20), rgba(245, 210, 122, 0.08)) !important;
            color: #f5d27a !important;
            border: 1px solid rgba(212, 175, 55, 0.35) !important;
        }

        /* Remove ugly bottom line */
        button[data-baseweb="tab"][aria-selected="true"]::after {
            background-color: transparent !important;
        }

        /* Better spacing inside tab content */
        div[data-testid="stTabs"] div[role="tabpanel"] {
            padding: 0.5rem 0.2rem 0.2rem 0.2rem;
        }

        /* Chart title spacing */
        div[data-testid="stTabs"] h4 {
            margin-top: 0.2rem;
            margin-bottom: 1rem;
            color: #ffffff;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# CACHE ORCHESTRATOR
# =========================

@st.cache_resource
def get_orchestrator() -> ProductReviewOrchestrator:
    """
    Cache the orchestrator to avoid recreating agents at every Streamlit refresh.
    """
    return ProductReviewOrchestrator()


# =========================
# UI HELPER FUNCTIONS
# =========================

def render_hero():
    """
    Display the top hero section.
    """
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Multi-Agent AI System · CrewAI · DistilBERT · Gemini</div>
            <div class="hero-title">AI Product Review Intelligence</div>
            <div class="hero-subtitle">
                Analyze product reviews with a multi-agent architecture, sentiment intelligence,
                business insights, human validation and automated report generation.
            </div>
            <br>
            <span class="pill">Collector Agent</span>
            <span class="pill">Sentiment Agent</span>
            <span class="pill">Insight Agent</span>
            <span class="pill">Report Agent</span>
            <span class="pill">Human-in-the-loop</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():
    """
    Display sidebar controls and return selected values.
    """
    with st.sidebar:
        st.markdown("## Control Center")
        st.caption("Configure the review analysis workflow.")

        st.divider()

        st.markdown("### Review Source")

        source_choice = st.radio(
            "Choose review source",
            [
                "Amazon Reviews 2023 product-level",
                "Hugging Face lightweight reviews",
                "Demo CSV"
            ]
        )

        if source_choice == "Amazon Reviews 2023 product-level":
            review_source = "amazon2023"
            st.warning("Real product-level review matching.")
        elif source_choice == "Hugging Face lightweight reviews":
            review_source = "huggingface"
            st.warning("Keyword-based Amazon review collection.")
        else:
            review_source = "sample_csv"
            st.warning("Stable local demo fallback.")

        st.divider()

        st.markdown("### Analysis Settings")

        max_reviews = st.slider(
            "Maximum reviews to analyze",
            min_value=3,
            max_value=20,
            value=5
        )

        st.divider()

    return review_source, max_reviews


def render_input_panel():
    """
    Display product input and manual reviews area.
    """
    st.markdown('<div class="section-title">Launch Product Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter a product name and optionally paste your own reviews.</div>',
        unsafe_allow_html=True
    )

    product_name = st.text_input(
        "Product name",
        placeholder="Example: iPhone case, Samsung Galaxy, headphones"
    )

    raw_reviews = st.text_area(
        "Optional: paste your own reviews, one review per line",
        placeholder=(
            "Manual reviews have priority over all other sources.\n"
            "Example:\n"
            "The product is excellent and easy to use.\n"
            "The battery life is too short."
        ),
        height=140
    )

    return product_name, raw_reviews


def display_sentiment_metrics(insights):
    """
    Display sentiment percentages as modern metric cards.
    """
    percentages = insights["sentiment_percentages"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Positive", f"{percentages['positive']}%")
    col2.metric("Negative", f"{percentages['negative']}%")
    col4.metric("Reviews analyzed", insights["total_reviews"])


def display_sentiment_chart(insights):
    """
    Display sentiment distribution as a premium gold bar chart.
    """
    percentages = insights["sentiment_percentages"]

    chart_df = pd.DataFrame(
        {
            "Sentiment": ["Negative", "Positive"],
            "Percentage": [
                percentages["negative"],
                percentages["positive"],
            ],
        }
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar(
            cornerRadiusTopLeft=8,
            cornerRadiusTopRight=8,
            color="#D4AF37"
        )
        .encode(
            x=alt.X(
                "Sentiment:N",
                title=None,
                axis=alt.Axis(
                    labelColor="#F5F5F5",
                    labelFontSize=13,
                    labelAngle=0
                )
            ),
            y=alt.Y(
                "Percentage:Q",
                title=None,
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(
                    labelColor="#F5F5F5",
                    labelFontSize=12,
                    gridColor="rgba(212, 175, 55, 0.15)"
                )
            ),
            tooltip=["Sentiment", "Percentage"]
        )
        .properties(
            height=340
        )
        .configure_view(
            strokeWidth=0
        )
        .configure_axis(
            domainColor="rgba(212, 175, 55, 0.25)",
            tickColor="rgba(212, 175, 55, 0.25)"
        )
        .configure(
            background="transparent"
        )
    )

    st.altair_chart(chart, use_container_width=True)


def display_sentiment_progress(insights):
    """
    Display sentiment percentages with progress bars.
    """
    percentages = insights["sentiment_percentages"]

    st.markdown("**Positive reviews**")
    st.progress(int(percentages["positive"]))

    st.markdown("**Negative reviews**")
    st.progress(int(percentages["negative"]))



def display_insight_cards(insights):
    """
    Display complaints and strengths in professional cards.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Common Complaints</div>
                <div class="section-subtitle">Main negative patterns detected by the agents.</div>
            """,
            unsafe_allow_html=True
        )

        complaints = insights.get("common_complaints", [])

        if complaints:
            for complaint in complaints:
                st.markdown(f"- {complaint}")
        else:
            st.info("No major complaint detected.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Product Strengths</div>
                <div class="section-subtitle">Main positive patterns detected by the agents.</div>
            """,
            unsafe_allow_html=True
        )

        strengths = insights.get("strengths", [])

        if strengths:
            for strength in strengths:
                st.markdown(f"- {strength}")
        else:
            st.info("No major strength detected.")

        st.markdown("</div>", unsafe_allow_html=True)


def display_review_table(analyzed_reviews):
    """
    Display analyzed reviews in a clean table.
    """
    table_df = pd.DataFrame(
        [
            {
                "Matched product": review.get("matched_product_title", review.get("product", "")),
                "Review": review.get("text", ""),
                "Sentiment": review.get("sentiment_label", ""),
                "Score": round(float(review.get("sentiment_score", 0)), 3),
                "Rating": review.get("rating", ""),
                "Model used": review.get("model_used", "unknown"),
                "Source": review.get("source", "unknown"),
            }
            for review in analyzed_reviews
        ]
    )

    st.dataframe(table_df, use_container_width=True, hide_index=True)


def display_review_cards(analyzed_reviews):
    """
    Display analyzed reviews as compact cards.
    """
    for index, review in enumerate(analyzed_reviews, start=1):
        sentiment = str(review.get("sentiment_label", "unknown")).lower()
        score = round(float(review.get("sentiment_score", 0)), 3)
        source = review.get("source", "unknown")
        model_used = review.get("model_used", "unknown")
        text = review.get("text", "")

        sentiment_class = ""

        if sentiment == "positive":
            sentiment_class = "positive-pill"
        elif sentiment == "negative":
            sentiment_class = "negative-pill"

        st.markdown(
            f"""
            <div class="glass-card">
                <span class="pill {sentiment_class}">{sentiment.upper()}</span>
                <span class="pill">Score: {score}</span>
                <span class="pill">Source: {source}</span>
                <span class="pill">Model: {model_used}</span>
                <p style="margin-top: 0.8rem; color: #e5e7eb;">
                    <strong>Review {index}:</strong> {text}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


def display_logs_panel():
    """
    Display JSON logs.
    """
    with st.expander("View JSON Logs", expanded=False):
        logs = read_logs(limit=50)

        if logs:
            st.json(logs)
        else:
            st.write("No logs yet.")


# =========================
# MAIN APP
# =========================

def main():
    """
    Main Streamlit application.
    """
    orchestrator = get_orchestrator()

    review_source, max_reviews = render_sidebar()
    render_hero()

    product_name, raw_reviews = render_input_panel()

    analyze_button = st.button("Analyze Reviews", type="primary", use_container_width=True)

    if analyze_button:
        try:
            if not product_name.strip():
                st.error("Please enter a product name.")
                return

            with st.spinner("Collector, Sentiment and Insight agents are analyzing reviews..."):
                analysis_result = orchestrator.run_analysis(
                    product_name=product_name,
                    max_reviews=max_reviews,
                    raw_reviews=raw_reviews,
                    review_source=review_source
                )

            st.session_state["analysis_result"] = analysis_result

            if "report_result" in st.session_state:
                del st.session_state["report_result"]

            st.success("Analysis completed. Human approval is required before final report generation.")

        except ProductReviewError as error:
            st.error(f"Application error: {error}")

        except Exception as error:
            st.error(f"Unexpected error: {error}")

    if "analysis_result" in st.session_state:
        result = st.session_state["analysis_result"]
        insights = result["insights"]

        st.markdown("---")

        st.markdown('<div class="section-title">Sentiment Intelligence Dashboard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Overview of customer perception and review distribution.</div>',
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.warning(f"Review source used: {result.get('review_source', 'unknown')}")

        with col_b:
            st.warning(f"Total reviews analyzed: {insights['total_reviews']}")

        display_sentiment_metrics(insights)

        tab1, tab2, tab3 = st.tabs(
            [
                "Sentiment Distribution",
                "Business Insights",
                "Analyzed Reviews"
            ]
        )

        with tab1:
            col_chart, col_progress = st.columns([2, 1])

            with col_chart:
                st.markdown("#### Sentiment chart")
                display_sentiment_chart(insights)

            with col_progress:
                st.markdown("#### Sentiment balance")
                display_sentiment_progress(insights)

        with tab2:
            display_insight_cards(insights)

        with tab3:
            view_mode = st.radio(
                "Review display mode",
                ["Table view", "Card view"],
                horizontal=True
            )

            if view_mode == "Table view":
                display_review_table(result["analyzed_reviews"])
            else:
                display_review_cards(result["analyzed_reviews"])

        st.markdown("---")

        st.markdown('<div class="section-title">Human-in-the-loop Checkpoint</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">The report agent can only generate the final report after human validation.</div>',
            unsafe_allow_html=True
        )

        st.warning("Please review the insights before approving final report generation.")

        human_approved = st.checkbox(
            "I reviewed the insights and approve final report generation."
        )

        generate_report_button = st.button(
            "Generate Final Report",
            use_container_width=True
        )

        if generate_report_button:
            try:
                with st.spinner("Report Agent is generating the final business report..."):
                    report_result = orchestrator.generate_report_after_approval(
                        analysis_result=result,
                        human_approved=human_approved
                    )

                st.session_state["report_result"] = report_result

                st.warning("Final report generated successfully.")

            except HumanApprovalRequiredError as error:
                st.error(str(error))

            except ProductReviewError as error:
                st.error(f"Report generation error: {error}")

            except Exception as error:
                st.error(f"Unexpected error: {error}")

    if "report_result" in st.session_state:
        st.markdown("---")

        st.markdown('<div class="section-title">Final Report</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Generated by the Report Agent after human approval.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="glass-card">
            """,
            unsafe_allow_html=True
        )

        st.markdown(st.session_state["report_result"]["report_text"])

        st.markdown("</div>", unsafe_allow_html=True)

        st.warning(
            f"Report saved at: {st.session_state['report_result']['report_path']}"
        )

    st.markdown("---")
    display_logs_panel()


if __name__ == "__main__":
    main()