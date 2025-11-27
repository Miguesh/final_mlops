import boto3
import datetime as dt
import os

import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# Configura tus datos
BUCKET_NAME = "mash-0124-mlops"
LOCAL_FILE = "response_1764208304973.json"  

today = dt.datetime.now()
time_stamp = today.strftime("%Y%m%d_%H%M%S")

file_name = os.path.basename(LOCAL_FILE)

# Fecha actual para organizar carpetas
today = dt.datetime.now()
key = f"origen_datos1/{time_stamp}_{file_name}"


# Conecta con S3 (usa tus credenciales de aws configure)
s3 = boto3.client("s3")

# Sube el archivo
s3.upload_file(LOCAL_FILE, BUCKET_NAME, key)

print(f"✅ Archivo subido a s3://{BUCKET_NAME}/{key}")