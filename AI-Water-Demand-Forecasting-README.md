# AI Water Demand Forecasting

**Predictive demand and pumping optimization for a campus water system.**

An AI-driven decision support system that uses XGBoost to forecast daily campus water demand and recommend pumping schedules, with the goal of lowering electricity costs and CO2 emissions.

**Live demo:** [AI Water Demand Forecasting on Streamlit](https://ai-water-demand-forecasting-ftcqsughf5xruy7ndwumkn.streamlit.app/)

## Project Overview

The campus faces water shortages and high electricity costs. This project uses an XGBoost regression model to forecast daily water demand and turns raw IoT sensor data into a dashboard that facility managers can act on.

## Key Features

- **Temporal analysis:** Takes the academic calendar (exams and vacations) and weekends into account.
- **Weather sensitivity:** Adjusts demand forecasts for temperature.
- **Cost optimization:** Gives pumping recommendations based on off-peak electricity tariffs.
- **Decision support:** Raises an alert when tank levels are predicted to fall below 25%.

## Tech Stack

- **Language:** Python 3.11
- **Model:** XGBoost (regression)
- **Dashboard:** Streamlit
- **Data processing:** Pandas, NumPy, Scikit-learn
- **Workflow:** Jupyter Notebook

## Project Files

| File or folder | Purpose |
| --- | --- |
| `01_Data_Transformation.ipynb` | Notebook that transforms the raw data |
| `processed_data/` | Processed data used by the model and the dashboard |
| `app.py` | The Streamlit dashboard |
| `requirements.txt` | Python dependencies |

## Getting Started

### Prerequisites

- Python 3.11

### Installation

1. Clone the repository and open the folder:

   ```bash
   git clone https://github.com/sharmi-panda/AI-Water-Demand-Forecasting.git
   cd AI-Water-Demand-Forecasting
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

Streamlit prints a local address (usually `http://localhost:8501`). Open it in your browser.
