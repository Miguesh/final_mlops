
import joblib
import pandas as pd

model = joblib.load('pkl/model_rf.pkl')

data = pd.DataFrame([1.0905917949529544,0,0,0,1,0,0,1,0,0,1,0]).T


pred = model.predict(data)
print(pred)
