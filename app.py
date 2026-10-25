import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Churn AI | Customer Prediction",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# 2. DARK THEME
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #080B12;
    color: #F8FAFC;
    font-family: Inter, sans-serif;
}
[data-testid="stHeader"] {
    background: #080B12;
}
.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #F8FAFC !important;
}
p, label {
    color: #CBD5E1;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg,#151E30,#0D1320);
    border: 1px solid #29364D;
    border-radius: 14px;
    padding: 18px;
}
[data-testid="stMetricValue"] {
    color: #60A5FA;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8;
}
div[data-testid="stForm"] {
    background: #0D1320;
    border: 1px solid #29364D;
    border-radius: 16px;
    padding: 22px;
}
div[data-testid="stPlotlyChart"] {
    background: #0D1320;
    border: 1px solid #29364D;
    border-radius: 14px;
    padding: 8px;
}
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 700;
    width: 100%;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    color: white;
    border: 1px solid #93C5FD;
}
hr {
    border-color: #29364D;
}
</style>
""", unsafe_allow_html=True)

BLUE = "#60A5FA"
RED = "#F87171"
PURPLE = "#A78BFA"
GREEN = "#34D399"
PANEL = "#111827"


# ==========================================
# 3. CHART STYLE
# ==========================================
def style_chart(fig, height=350):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color="#F8FAFC", family="Inter"),
        title_font=dict(size=17),
        margin=dict(l=20, r=25, t=65, b=25),
        height=height
    )
    return fig


# ==========================================
# 4. HEADER
# ==========================================
st.markdown("""
<div style="padding:10px 0 20px 0;">
    <div style="
        color:#60A5FA;
        font-size:12px;
        font-weight:800;
        letter-spacing:3px;">
        MACHINE LEARNING PROJECT
    </div>
    <div style="
        color:#F8FAFC;
        font-size:40px;
        font-weight:800;
        margin-top:10px;">
        Customer Churn
        <span style="color:#60A5FA;">Prediction</span>
    </div>
    <div style="color:#94A3B8;margin-top:10px;">
        Predict customer retention and view profile summary.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ==========================================
# 5. LOAD SAVED MODEL
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "customer_churn_model.joblib"
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Put customer_churn_model.joblib "
        "in the same folder as app.py."
    )
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    saved_data = load_model(MODEL_PATH)
    model = saved_data["model"]
    threshold = float(saved_data.get("threshold", 0.60))
except Exception as e:
    st.error(f"Could not load the saved model: {e}")
    st.stop()


# ==========================================
# 6. CUSTOMER INPUT FORM
# ==========================================
st.markdown("## 👤 Customer Information")
st.write("Enter the customer details, then click Predict Churn.")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])

        senior = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        tenure = st.number_input(
            "Tenure (Months)", 0, 72, 12
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

    with col3:
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.markdown("### 💳 Billing Information")

    bill1, bill2 = st.columns(2)

    with bill1:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=2000.0,
            value=70.0,
            step=1.0
        )

    with bill2:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=840.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "🚀 Predict Churn"
    )


# ==========================================
# 7. SHOW RESULTS ONLY AFTER BUTTON CLICK
# ==========================================
if submitted:

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        if 1 in classes:
            yes_index = classes.index(1)
        elif "Yes" in classes:
            yes_index = classes.index("Yes")
        else:
            st.error("Could not identify the Churn class.")
            st.stop()

        churn_probability = float(probabilities[yes_index])
        stay_probability = 1 - churn_probability
        prediction = churn_probability >= threshold

        # ==========================================
        # 8. INDIVIDUAL PREDICTION
        # ==========================================
        st.divider()
        st.markdown("## 🎯 Prediction Result")

        result_col, probability_col = st.columns([1, 1.2])

        with result_col:
            if prediction:
                st.error("⚠️ Higher Churn Risk")
                st.markdown(
                    "### Likely to Leave\n\n"
                    "The model predicts that this customer is at risk."
                )
            else:
                st.success("✅ Lower Churn Risk")
                st.markdown(
                    "### Likely to Stay\n\n"
                    "The model predicts that this customer is less likely to leave."
                )

            st.metric(
                "Estimated Churn Probability",
                f"{churn_probability:.1%}"
            )

            st.progress(
                min(max(churn_probability, 0.0), 1.0),
                text="Churn Probability"
            )

        with probability_col:
            probability_df = pd.DataFrame({
                "Outcome": ["Stay", "Churn"],
                "Probability": [
                    stay_probability * 100,
                    churn_probability * 100
                ]
            })

            fig = px.bar(
                probability_df,
                x="Outcome",
                y="Probability",
                color="Outcome",
                text="Probability",
                color_discrete_map={
                    "Stay": BLUE,
                    "Churn": RED
                },
                title="Predicted Customer Outcome"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                cliponaxis=False
            )

            fig.update_layout(
                showlegend=False,
                yaxis_title="Probability (%)",
                yaxis_range=[0, 115]
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        # ==========================================
        # 9. CUSTOMER ANALYTICS (SUMMARY METRICS)
        # ==========================================
        st.divider()
        st.markdown("## 📊 Customer Analytics Summary")
        st.caption(
            "Overview of the entered customer profile metrics."
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Customer Tenure", f"{tenure} Months")
        m2.metric("Monthly Charges", f"${monthly_charges:.2f}")
        m3.metric("Selected Contract", contract)
        m4.metric("Model Churn Status", "At Risk" if prediction else "Stable")

        st.info(
            f"💡 **Profile Summary:** This customer has a tenure of **{tenure} months** "
            f"and a monthly bill of **${monthly_charges:.2f}** under a **{contract}** contract, "
            f"resulting in an estimated churn probability of **{churn_probability:.1%}**."
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")


# ==========================================
# 10. FOOTER
# ==========================================
st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#64748B;
    padding:15px;
    font-size:12px;">
    Customer Churn Prediction Dashboard
    <br>
    Streamlit · Plotly · Machine Learning
</div>
""", unsafe_allow_html=True)