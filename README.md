# Healthcare-Resource-Optimisation-App
Developed a Healthcare Resource Optimization App to forecast medical resource needs and optimize their allocation during health crises like pandemics.
# 🏥 Healthcare Resource Optimization App

A time-series based forecasting and resource planning tool developed to help hospitals, healthcare organizations, and government bodies optimize the allocation of medical resources (beds, ventilators, oxygen cylinders, etc.) during healthcare emergencies such as pandemics.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model & Algorithms](#model--algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Use Cases](#use-cases)
- [Results](#results)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

This project forecasts the demand for essential healthcare resources using historical health data. It applies advanced time series modeling to provide precise predictions, enabling smarter and more efficient medical logistics management.

---

## ✨ Features

- 📈 Forecasts demand for ICU beds, ventilators, oxygen, and staff
- ⏱️ Real-time predictive analytics
- 📊 Dynamic and interactive visualizations
- 🧠 Advanced time-series model using Facebook Prophet
- 🧪 Accurate uncertainty intervals for critical planning
- 🛡️ Secure and scalable backend
- 🚑 Useful for hospitals, NGOs, and policymakers

---

## 🛠️ Tech Stack

- **Frontend**: Python (CLI or Streamlit for UI)
- **Backend**: Python
- **Forecasting Model**: [Facebook Prophet](https://facebook.github.io/prophet/)
- **Bayesian Modeling**: PyStan
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Environment**: venv, pip

---

## 📊 Model & Algorithms

- **Prophet Model** (by Facebook):
  - Additive model including trend, seasonality, and holiday effects.
  - Handles missing data, outliers, and sudden changes in trend well.

- **PyStan**:
  - Probabilistic programming language used as the backend of Prophet.
  - Bayesian inference ensures credible intervals for forecasting.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/sd45saswat/healthcare-resource-optimization-app.git
cd healthcare-resource-optimization-app
