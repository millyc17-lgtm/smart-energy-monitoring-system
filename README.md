# Smart Energy Monitoring System

A dashboard for exploring household electricity consumption, built with Python, SQLite and Streamlit.

## Data
Low Carbon London smart meter data (Kaggle, "Smart meters in London"), using `block_0.csv` (50 households, half-hourly kWh). Readings run from December 2011 to February 2014, so this is historical data, not live usage. Only one household reported before March 2012, so trend analysis uses October 2012 onwards.

## Run it
1. `pip install -r requirements.txt`
2. Put `block_0.csv` in `data/raw/`
3. `python clean.py`, then `python features.py`, then `python build_db.py`
4. `streamlit run dashboard/app.py`

## Limitations
- Small sample of 50 households, not representative of the UK average
- The electricity price is an adjustable assumption, not a real tariff