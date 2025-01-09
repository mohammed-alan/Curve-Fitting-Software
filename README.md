# Curve Fitting Tool

## Functionality

This web-based tool allows users to perform curve fitting on data. You can either manually input `x` and `y` values or upload a CSV file. Once the data is loaded, choose from various curve types (Linear, Polynomial, Exponential, Gaussian, or Histogram) to fit the data. The tool visualizes the data along with the fitted curve and displays the equation. It also calculates the Mean Absolute Error (MAE) and R-squared value to evaluate the fit. You can download the fitted data as a CSV file and the plot as a PNG.

### Features:
- Manual or CSV data input
- Multiple curve fitting options (Linear, Polynomial, Exponential, Gaussian)
- Visualization of the data and fitted curve
- Display of fit equation and quality metrics (MAE, R-squared)
- Download options for fitted data and plot

---

## Setup Instructions

To run the app locally, follow these steps:

### 1. Install dependencies:

First, ensure you have Python 3.7 or higher installed. Then, install the required libraries:

```bash
xpip install streamlit pandas numpy matplotlib scipy
```
### 2. Run the Streamlit app:

After installing the dependencies, navigate to the folder containing the script and run:
```bash
streamlit run app.py
```

### 3. Access the tool:
Once the app starts, open your browser and go to http://localhost:8501 to interact with the tool.
