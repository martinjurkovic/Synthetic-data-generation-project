# Download Biodegradability and Mutagenesis
python src/data/download_datasets.py

# Airbnb
# the download does not work for some reason, download manually from:
# https://www.kaggle.com/competitions/airbnb-recruiting-new-user-bookings/data

# Telstra
kaggle datasets download -d yifanxie/telstra-competition-dataset 
mv telstra-competition-dataset.zip data/telstra-competition-dataset.zip

# Rossman Store Sales
kaggle datasets download -d pratyushakar/rossmann-store-sales
mv rossmann-store-sales.zip data/rossmann-store-sales.zip

# Coupon Purchase Prediction
kaggle competitions download -c coupon-purchase-prediction
mv coupon-purchase-prediction.zip data/coupon-purchase-prediction.zip

# World Development Indicators
kaggle datasets download -d theworldbank/world-development-indicators
mv world-development-indicators.zip data/world-development-indicators.zip

python src/data/extract_datasets.py