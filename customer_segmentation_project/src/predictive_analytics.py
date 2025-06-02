from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_predictive_model(data, target_column, segment_column):
    """
    Train a predictive model for each customer segment.
    """
    models = {}
    for segment in data[segment_column].unique():
        segment_data = data[data[segment_column] == segment]
        X = segment_data.drop([target_column, segment_column], axis=1)
        y = segment_data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Segment {segment} - Mean Squared Error: {mse}')
        
        models[segment] = model
    
    return models

def predict_future_behavior(models, new_data, segment_column):
    """
    Predict future behavior for new customer data.
    """
    predictions = []
    for _, row in new_data.iterrows():
        segment = row[segment_column]
        features = row.drop(segment_column).values.reshape(1, -1)
        prediction = models[segment].predict(features)
        predictions.append(prediction[0])
    
    return predictions
