import shutil
import os
import kagglehub

# this downloads data to cache, so we store in directory instead
path_hourly_consumption = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
path_consumption_prices_weather = kagglehub.dataset_download("nicholasjhana/energy-consumption-generation-prices-and-weather")
path_smart_home = kagglehub.dataset_download("mexwell/smart-home-energy-consumption")


working_dir = os.path.join(os.getcwd(), "raw")
os.makedirs(working_dir, exist_ok=True)

shutil.copytree(path_hourly_consumption, os.path.join(working_dir, "hourly-energy-consumption"), dirs_exist_ok=True)
shutil.copytree(path_consumption_prices_weather, os.path.join(working_dir, "energy-prices-weather"), dirs_exist_ok=True)
shutil.copytree(path_smart_home, os.path.join(working_dir, "smart_home"), dirs_exist_ok=True)

print("Datasets copied to:", working_dir)