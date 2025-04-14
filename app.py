import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import pickle
import streamlit as st  # type: ignore
import urllib.parse
import webbrowser
from prophet import Prophet  # type: ignore
from sklearn.ensemble import IsolationForest  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from pyomo.environ import ( # type: ignore
    ConcreteModel, Var, Objective, Constraint, NonNegativeReals, minimize, SolverFactory, ConstraintList
)

# ------------------- Helper Functions ----------------------

def preprocess_data(file):
    data = pd.read_csv(file)
    if 'timestamp' not in data.columns:
        st.error("CSV must contain a 'timestamp' column.")
        st.stop()

    data['timestamp'] = pd.to_datetime(data['timestamp'], dayfirst=True)
    data.set_index('timestamp', inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    return data, data_scaled

def detect_anomalies(data_scaled, column):
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(data_scaled[[column]])
    data_scaled['anomaly'] = clf.predict(data_scaled[[column]])
    data_scaled['anomaly'] = data_scaled['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=data_scaled, x=data_scaled.index, y=column, hue='anomaly', ax=ax, palette={"Normal": "blue", "Anomaly": "red"})
    ax.set_title(f"Anomaly Detection in {column}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def forecast_resource_demand(data, column, forecast_period=365 * 20):
    df = data.reset_index()[['timestamp', column]].dropna()
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_period, freq='D')
    forecast = model.predict(future)

    fig = model.plot(forecast)
    plt.title(f"Forecast for {column} (Next {forecast_period // 365} Years)", fontsize=14)
    return fig, model, forecast

def load_forecast_model(model_file, forecast_period=365 * 20):
    try:
        model = pickle.load(model_file)
        future = model.make_future_dataframe(periods=forecast_period, freq='D')
        forecast = model.predict(future)

        fig = model.plot(forecast)
        plt.title("Loaded Forecast Model Output", fontsize=14)
        return fig
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def optimize_allocation(staff, demand):
    departments = list(staff.keys())
    patient_severity = {'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0}
    patient_demand = {'Critical': 10, 'High': 8, 'Medium': 5, 'Low': 2}

    model = ConcreteModel()
    model.x = Var(departments, domain=NonNegativeReals)
    model.dev = Var(departments, domain=NonNegativeReals)  # absolute deviation variables

    model.obj = Objective(
        expr=sum(model.dev[d] for d in departments),
        sense=minimize
    )

    model.dev_constraints = ConstraintList() # type: ignore
    for d in departments:
        model.dev_constraints.add(model.x[d] - demand[d] <= model.dev[d])
        model.dev_constraints.add(demand[d] - model.x[d] <= model.dev[d])

    model.total_constraint = Constraint(
        expr=sum(model.x[d] for d in departments) <= sum(staff.values())
    )

    if 'ICU' in departments:
        icu_requirement = sum(patient_severity[sev] * patient_demand[sev] for sev in patient_severity)
        model.icu_constraint = Constraint(expr=model.x['ICU'] >= icu_requirement)

    glpk_path = r"C:\Users\singh\HEALTHCARE_PROJECT\glpk\glpk-4.65\w64\glpsol.exe"
    solver = SolverFactory('glpk', executable=glpk_path)

    if solver is None or not solver.available():
        st.error(f"GLPK solver not found or not executable at:\n{glpk_path}")
        return None, None

    try:
        result = solver.solve(model, tee=False)

        if str(result.solver.status).lower() != 'ok' or str(result.solver.termination_condition).lower() != 'optimal':
            st.warning("Solver did not reach an optimal solution.")
            return None, None

        allocation = {d: float(model.x[d].value) for d in departments}
        icu_allocation = allocation.get('ICU', 0.0)
        return allocation, icu_allocation

    except Exception as e:
        st.error(f"Solver Error: {e}")
        return None, None


# ------------------- Streamlit App ----------------------

st.set_page_config(page_title="Healthcare Resource Planner", layout="wide")
st.markdown("<h1 style='color: #FF2DF1;'>🏥 Healthcare Resource Analytics & ICU Optimization</h1>", unsafe_allow_html=True)
st.markdown("""
    <style>
    body {
        background-color: #e0ffff;
    }
    .stApp {
        background-image: linear-gradient(to bottom right, #e0ffff, #ccffff);
        background-size: cover;
    }
    </style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your healthcare CSV file", type=['csv'])

if uploaded_file:
    data, data_scaled = preprocess_data(uploaded_file)
    st.success("✅ Data loaded and preprocessed successfully!")
    st.write("### Data Preview", data.head())

    resource = st.selectbox("Select resource for anomaly detection & forecasting:", data.columns)

    st.markdown("---")
    st.subheader("🔍 Anomaly Detection")
    fig_anomaly = detect_anomalies(data_scaled, resource)
    st.pyplot(fig_anomaly)

    st.markdown("---")
    st.subheader("📈 Forecasting")

    forecast_method = st.radio("Choose Forecasting Option", ["Train new model", "Load saved model (.pkl)"])

    if forecast_method == "Train new model":
        forecast_period = st.slider("Forecast Period (in days)", min_value=30, max_value=7300, value=730)
        if st.button("Train & Forecast"):
            fig_forecast, model, forecast = forecast_resource_demand(data, resource, forecast_period)
            st.pyplot(fig_forecast)

            with open(f"{resource}_forecast_model.pkl", "wb") as f:
                pickle.dump(model, f)
            st.success(f"✅ Model saved as `{resource}_forecast_model.pkl`")

    else:
        model_file = st.file_uploader("Upload .pkl model", type=['pkl'])
        if model_file and st.button("Load & Forecast"):
            fig_model = load_forecast_model(model_file)
            if fig_model:
                st.pyplot(fig_model)

    st.markdown("---")
    st.subheader("⚙️ ICU Resource Optimization")
    with st.form("icu_form"):
        departments = ['ICU', 'Emergency', 'DayShift', 'NightShift', 'Medicine', 'Ventilation']
        staff = {dept: st.number_input(f"{dept} staff available", min_value=0.0, value=10.0) for dept in departments}
        demand = {dept: st.number_input(f"{dept} demand", min_value=0.0, value=20.0) for dept in departments}
        submitted = st.form_submit_button("Optimize Allocation")

    if submitted:
        allocation, icu_allocation = optimize_allocation(staff, demand)
        if allocation:
            st.success("✅ Optimized Allocation Successful")
            st.dataframe(pd.DataFrame.from_dict(allocation, orient='index', columns=["Allocated"]).round(2))

st.markdown("---")
st.markdown("<h3 style='color: purple;'>🔍 Explore Solution Options</h3>", unsafe_allow_html=True)
st.write("If you're facing issues or need help with your problem, try one of the following options:")

query_text = "how to optimize healthcare resource allocation using Pyomo"
query = urllib.parse.quote(query_text)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Search on Google"):
        webbrowser.open_new_tab(f"https://www.google.com")

with col2:
    if st.button("💬 Ask ChatGPT"):
        webbrowser.open_new_tab("https://chat.openai.com/chat")

with col3:
    if st.button("💡 Stack Overflow"):
        webbrowser.open_new_tab(f"https://stackoverflow.com")

st.markdown("<h3 style='color: purple;'>🏥 🎓 Get to know more about Healthcare in long term</h3>", unsafe_allow_html=True)

healthcare_sites = {
    "Statista": "https://www.statista.com/",
    "IBISWorld": "https://www.ibisworld.com/",
    "Frost & Sullivan": "https://www.frost.com/",
    "GlobalData Healthcare": "https://www.globaldata.com/store/industry/healthcare/",
    "Crunchbase": "https://www.crunchbase.com/",
    "AngelList": "https://angel.co/",
    "MedCity News": "https://medcitynews.com/",
    "Rock Health": "https://rockhealth.com/",
    "FDA": "https://www.fda.gov/",
    "HIPAA Journal": "https://www.hipaajournal.com/",
    "EMA": "https://www.ema.europa.eu/en",
    "HealthIT.gov": "https://www.healthit.gov/",
    "HIMSS": "https://www.himss.org/",
    "Nature Digital Medicine": "https://www.nature.com/npjdigitalmed/",
    "BioPharmGuy": "https://www.biopharmguy.com/",
    "Medica Trade Fair": "https://www.medica-tradefair.com/",
    "MassBio": "https://www.massbio.org/",
    "Coursera": "https://www.coursera.org/",
    "edX": "https://www.edx.org/",
    "FutureLearn": "https://www.futurelearn.com/",
    "CB Insights": "https://www.cbinsights.com/",
    "PitchBook": "https://pitchbook.com/",
    "Tracxn": "https://tracxn.com/",
    "WHO": "https://www.who.int/",
    "World Bank Health": "https://www.worldbank.org/en/topic/health",
    "UNDP Health": "https://www.undp.org/health"
}

selected_site = st.selectbox("Choose a site to explore:", list(healthcare_sites.keys()))

if selected_site:
    st.markdown(f"[Click here to visit {selected_site}]({healthcare_sites[selected_site]})")