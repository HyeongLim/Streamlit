import numpy as np
import pandas as pd
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from scipy import stats
from scipy.stats import gaussian_kde

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"


@st.cache_data
def load_posterior():
    return az.from_netcdf("posterior.nc")


def compute_predicted_log_lambda_with_no_uncertain_fe_content(posterior, fe_content, seed=42):
    a = np.exp(posterior.posterior.log_a.values.flatten())
    mu = a * np.log(fe_content) + posterior.posterior.b.values.flatten()

    np.random.seed(seed)
    return (
        stats.norm.rvs(
            size=posterior.posterior.sizes["chain"]
            * posterior.posterior.sizes["draw"]
        )
        * posterior.posterior.sigma_regression_error.values.flatten()
        + mu
    )


def compute_predicted_log_lambda_with_uncertain_fe_content(posterior, fe_content, seed=9999):
    mu = (
        np.exp(posterior.posterior.log_a.values.flatten())
        * np.log(fe_content)
        + posterior.posterior.b.values.flatten()
    )

    np.random.seed(seed)
    return (
        stats.norm.rvs(
            size=posterior.posterior.sizes["chain"]
            * posterior.posterior.sizes["draw"]
        )
        * posterior.posterior.sigma_log_lambda_measured_log_fe.values.flatten()
        + mu
    )


def predict_and_plot(fe_content, porosity, solids_density):
    posterior = load_posterior()

    log_lambda = compute_predicted_log_lambda_with_uncertain_fe_content(
        posterior, fe_content
    )

    sl_ratio = ((1 - porosity) / porosity) * (solids_density * 1000)
    log_lambda = np.log(np.exp(log_lambda) * sl_ratio)

    data = np.asarray(log_lambda)
    mean_val = data.mean()
    std_val = data.std()

    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 1000)
    y_vals = kde(x_vals)

    q15, q85 = np.percentile(data, [14, 87])

    stats_table = pd.DataFrame(
        {
            "Statistic (λ)": ["Mean", "Standard Deviation"],
            "Value [1/yr]": [
                np.exp(mean_val),
                np.exp(mean_val + std_val) - np.exp(mean_val),
            ],
        }
    )

    half_life = pd.DataFrame(
        [["Mean half-life (first-order)", 0.693 / np.exp(mean_val)]],
        columns=["", "[yr]"],
    )

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.hist(data, bins=30, alpha=0.6, edgecolor="black")
    ax1.set_xlabel(r"ln($\lambda$) [1/yr]")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Histogram of ln(λ)")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.kdeplot(data, ax=ax2)
    mask = (x_vals >= q15) & (x_vals <= q85)
    ax2.fill_between(x_vals[mask], y_vals[mask], alpha=0.3, label="15–85%")
    ax2.axvline(mean_val, linestyle="--", label="Mean")
    ax2.set_xlabel(r"ln($\lambda$) [1/yr]")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.set_title("PDF of ln(λ)")

    return fig1, fig2, stats_table, half_life


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Abiotic TCE Reduction Tool", layout="wide")

st.markdown("# Tool for Estimating First-Order Abiotic TCE Reduction Rate Constants in Anoxic Clay-Rich Groundwater Sediments")
          
st.markdown("""
     This tool was developed to help environmental professionals/practitioners estimate first-order abiotic reduction rate constants for trichloroethylene (TCE)
     in anoxic and clay-rich groundwater sediments. The tool is based on a linear regression model that relates the rate constant to the concentration of extractable 
     reactive ferrous iron (Fe(II)), determined using a mild hydrochloric acid (HCl) extraction method as described in [Schaefer et al. (2025)](https://doi.org/10.1111/gwmr.12709). 
    """)
st.markdown("""
     The model was developed using data compiled from multiple studies using clay-rich groundwater sediments for anoxic conditions. 
     To account for uncertainty in the input data and model predictions, the regression was implemented within a Bayesian statistical framework, per [Storiko et al. (2024)]( https://doi.org/10.1111/gwmr.12667). 
    """)
st.markdown("""
     Required input parameters are measured Fe(II) concentration, sediment porosity, and sediment solids density, all from a site. The tool returns both an estimate of the mean first-order rate constant (λ [in 1/year]) and its uncertainty.
     This rate constant can be used in fate and transport models to assess abiotic dechlorination potential in clay-rich sediments. 
    """)

st.image("paper_figure.png")

st.markdown("""
    Figure 1 above shows the data and fitted linear regression model that predicts the natural log of the rate constant (i.e., ln(λ) [in L/g/year]) as a function of the natural log of the measured Fe(II) content (i.e., ln(Fe(II) [in mg/kg of dry soil]).
    The rate constant (λ) used in Figure 1 is normalized by the solids-to-water ratio (i.e., solids density × (1/porosity – 1)). 
    """)
st.markdown("""
    - The **gray line** represents the median prediction of the natural log of the rate constant (ln(λ)) as a function of the natural log of the Fe(II) content.
    - The **shaded gray areas** represent the 90% and 50% credible intervals, showing the uncertainty in the regression.
    - **Blue crosses** represent posterior estimates for individual sediment samples (median and 5th–95th percentile range).
    - **Colored markers** show the actual measured Fe(II) values and associated rate constants reported in literature.             
    """)  
st.markdown("""
    For more technical detail on the model development, including data sources and code, refer to the full open-access publication: https://doi.org/10.1111/gwmr.12667
    """)    

with st.sidebar:
    st.header("Input Parameters")
    fe_content = st.number_input("Fe Content (mg/kg)", 1.0, value=2000.0)
    porosity = st.number_input("Porosity (0–1)", 0.01, 0.99, value=0.4)
    solids_density = st.number_input(
        "Solids Density (g/cm³)", 1.0, value=2.65
    )

run = st.button("Run Analysis")

if run:
    with st.spinner("Running Bayesian prediction…"):
        fig1, fig2, stats_df, half_life_df = predict_and_plot(
            fe_content, porosity, solids_density
        )

    st.subheader("Results")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)

    st.subheader("Summary Statistics")
    st.dataframe(stats_df, use_container_width=True)

    st.subheader("Half-life Estimate")
    st.dataframe(half_life_df, use_container_width=True)