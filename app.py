from fastapi import FastAPI 
from typing import List 
import numpy as np 
import pandas as pd 
from pydantic import BaseModel 
from statsmodels.tsa.deterministic import DeterministicProcess 
from sklearn.linear_model import LinearRegression 
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt 
import io 
import base64 
 
app = FastAPI() 
 
class DateRangeInput(BaseModel): 
    start_date: str 
    end_date: str 
 
@app.post("/predict/") 
async def predict_sales(date_range: DateRangeInput): 
    start_date = datetime.strptime(date_range.start_date, '%Y-%m-%d') 
    end_date = datetime.strptime(date_range.end_date, '%Y-%m-%d') 
     
    train_df = pd.read_csv(r'E:\retail\train_data.csv', parse_dates=['date']) 
    store_sales = train_df.copy() 
    store_sales = store_sales.set_index('date').to_period('D') 
    store_sales = store_sales.set_index(['state', 'category_of_product'], append=True) 
    average_sales = store_sales.groupby('date').mean()['sales'] 
 
    dp = DeterministicProcess( 
        index=average_sales.index, 
        constant=False, 
        order=3, 
        drop=True 
    ) 
    X = dp.in_sample() 
     
    # Generate the out-of-sample index for the specified date range 
    X_fore = dp.out_of_sample(steps=len(pd.date_range(start=start_date, end=end_date))) 
    X_fore.index = pd.date_range(start=start_date, end=end_date) 
 
    y = average_sales.copy() 
    model = LinearRegression() 
    model.fit(X, y) 
 
    y_fore = pd.Series(model.predict(X_fore), index=X_fore.index) 
 
    # Filter the predicted values within the specified date range 
    X_fore = X_fore[(X_fore.index >= start_date) & (X_fore.index <= end_date)] 
    y_fore = y_fore[(y_fore.index >= start_date) & (y_fore.index <= end_date)] 
 
    # Prepare data for bar chart 
    dates = [pred_date.strftime('%Y-%m-%d') for pred_date in X_fore.index] 
    predicted_sales = y_fore.tolist() 
 
    # Create and save the bar chart with improved layout 
    plt.figure(figsize=(10, 6)) 
    plt.bar(dates, predicted_sales) 
    step_size = max(1, len(dates) // 10) 
    plt.xticks(np.arange(0, len(dates), step_size), dates[::step_size], rotation=45, ha="right") 
    plt.tight_layout() 
     
    buffer = io.BytesIO() 
    plt.savefig(buffer, format="png") 
    plt.close() 
     
    buffer.seek(0) 
    image_base64 = base64.b64encode(buffer.read()).decode() 
 
    # Prepare response including the bar chart image 
    response = { 
        "predictions": [ 
            { 
                "date": dates[i], 
                "predicted_sales": predicted_sales[i] 
            } 
            for i in range(len(dates)) 
        ], 
        "chart_image": f"data:image/png;base64,{image_base64}" 
    } 
 
    return response 
 
if __name__ == "__main__": 
    import uvicorn 
 
    uvicorn.run(app, host="127.0.0.1", port=8000)