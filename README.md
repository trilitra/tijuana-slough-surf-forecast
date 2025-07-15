🌊 Tijuana Slough Surf Forecast
Tijuana Slough Surf Forecast is a Python-based tool designed to fetch and analyze surf forecasts for the Tijuana Slough area. It extracts surf data from external sources and processes it into a usable, human-readable format for surf enthusiasts, researchers, or hobbyist forecasters.

📌 Features
Automated data scraping from NOAA forecast pages

Text-based surf condition summaries

Location-specific forecast details for Tijuana Slough, CA

Simple, extensible Python codebase

🛠️ Requirements
This project is built with Python. To run it locally, ensure you have the following installed:

Python 3.8+

Poetry (recommended) or pip

PyCharm (for IDE support, optional)

📦 Installation
Clone the repository:

bash
Копировать
Редактировать
git clone https://github.com/trilitra/tijuana-slough-surf-forecast.git
cd tijuana-slough-surf-forecast
Option 1: Using Poetry (Recommended)
bash
Копировать
Редактировать
poetry install
poetry shell
Option 2: Using Pip
bash
Копировать
Редактировать
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
🧪 Running the Script
Once dependencies are installed, you can run the main script:

bash
Копировать
Редактировать
python surf_forecast.py
Or in PyCharm, right-click surf_forecast.py and select Run.

🧰 Project Structure
bash
Копировать
Редактировать
tijuana-slough-surf-forecast/
├── surf_forecast.py         # Main script
├── parser.py                # Forecast parsing logic
├── utils.py                 # Helper functions
├── requirements.txt         # Package requirements
└── README.md                # Project documentation
💡 Notes
Forecast data is pulled from NOAA and may change format unexpectedly.

This tool is intended for educational or personal use and not for professional marine navigation or safety-critical applications.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
