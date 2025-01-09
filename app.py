import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.set_page_config(page_title="Curve Fitting Tool", layout="wide")

st.markdown("""
    <style>
    .reportview-container {
        background-color: #f7f7f7;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        font-size: 16px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stTextInput, .stTextArea {
        font-size: 14px;
    }
    .stWrite {
        font-size: 16px;
    }
    .stSidebar, .stSelectbox {
        font-size: 16px;
    }
    .stMarkdown {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Curve Fitting Tool")

st.header("Step 1: Enter Your Data")
data_option = st.radio(
    "How would you like to enter the data?",
    ("Enter manually", "Upload CSV"),
    index=0
)

def create_sample_data():
    x = np.linspace(0, 10, 50)
    y = 2.5 * x + np.random.normal(0, 1, x.size)
    return x, y

curve_type = st.selectbox(
    "Choose the visualization type:",
    ["Linear", "Polynomial", "Exponential", "Gaussian", "Histogram"],
    index=0
)

if data_option == "Enter manually":
    st.info("Enter comma-separated values for x and y. Example: 1, 2, 3, 4, 5")
    x_values = st.text_area("X values:", "1, 2, 3, 4, 5")
    y_values = st.text_area("Y values:", "1.5, 2.8, 3.2, 4.1, 5.3")
    
    try:
        x = np.array([float(i) for i in x_values.split(",")])
        y = np.array([float(i) for i in y_values.split(",")])
        if len(x) != len(y) and curve_type != "Histogram":
            pass
    except ValueError:
        if curve_type == "Histogram":
            pass
        else:
            st.error("Please enter valid numeric values for both x and y.")

elif data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded data:")
        st.dataframe(df.head())  # Show first 5 rows of the uploaded CSV
        x_column = st.selectbox("Select the X column:", df.columns)
        y_column = st.selectbox("Select the Y column:", df.columns)
        x = df[x_column].values
        y = df[y_column].values
    else:
        st.write("Sample data preview:")
        x, y = create_sample_data()
        st.write(pd.DataFrame({'x': x, 'y': y}))

st.header("Step 2: Visualization")

def linear(x, a, b):
    return a * x + b

def polynomial(x, *params):
    return np.polyval(params, x)

def exponential(x, a, b):
    return a * np.exp(b * x)

def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

if 'x' in locals() and curve_type == "Histogram":
    st.subheader("Histogram")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(x, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
    ax.set_title("Histogram of X Values", fontsize=16)
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    st.pyplot(fig)

elif 'x' in locals() and 'y' in locals() and len(x) == len(y):
    if curve_type == "Linear":
        popt, _ = curve_fit(linear, x, y)
        fit_y = linear(x, *popt)
        equation = f"y = {popt[0]:.4f}x + {popt[1]:.4f}"
    elif curve_type == "Polynomial":
        degree = st.slider("Select polynomial degree:", 1, 10, 2)
        popt = np.polyfit(x, y, degree)
        fit_y = polynomial(x, *popt)
        equation = f"y = " + " + ".join([f"{coef:.4f}x^{deg}" for deg, coef in enumerate(popt[::-1])])
    elif curve_type == "Exponential":
        popt, _ = curve_fit(exponential, x, y)
        fit_y = exponential(x, *popt)
        equation = f"y = {popt[0]:.4f}e^({popt[1]:.4f}x)"
    elif curve_type == "Gaussian":
        popt, _ = curve_fit(gaussian, x, y)
        fit_y = gaussian(x, *popt)
        equation = f"y = {popt[0]:.4f}e^(-((x - {popt[1]:.4f})^2)/(2 * {popt[2]:.4f}^2))"

    st.subheader(f"{curve_type} Curve Fit")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, label="Data", color="blue", s=40)
    ax.plot(x, fit_y, label=f"Fitted {curve_type} Curve", color="red", linewidth=2)

    ax.text(0.05, 0.95, f"Equation: {equation}", transform=ax.transAxes, fontsize=14, verticalalignment='top', color='green')

    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.legend()
    st.pyplot(fig)

    residuals = y - fit_y
    mae = np.mean(np.abs(residuals))  # Mean Absolute Error (MAE)
    r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Fit Parameters:")
        st.markdown(f"**{popt}**")

    with col2:
        st.markdown("### Equation of the Fitted Curve:")
        st.markdown(f"**{equation}**")

    with col3:
        st.markdown("### Quality of Fit:")
        st.markdown(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.markdown(f"**R-squared:** {r_squared:.4f}")

    fitted_data = pd.DataFrame({"x": x, "y": y, "fitted_y": fit_y})
    def convert_to_csv(data):
        return data.to_csv(index=False).encode('utf-8')
    csv = convert_to_csv(fitted_data)
    st.download_button("Download Fitted Data as CSV", csv, "fitted_data.csv", "text/csv")

    plot_image = "curve_fit_plot.png"
    fig.savefig(plot_image)
    with open(plot_image, "rb") as file:
        st.download_button("Download Plot as PNG", file, plot_image, "image/png")

else:
    if curve_type != "Histogram":
        st.error("The number of x and y values must be the same for curve fitting.")
