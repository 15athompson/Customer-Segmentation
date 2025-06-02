import schedule
import time
from src.customer_segmentation import perform_segmentation
from src.data_loader import load_data

def update_model():
    """
    Function to update the segmentation model with new data.
    """
    print("Updating segmentation model...")
    data = load_data()  # Load new data
    perform_segmentation(data)  # Perform segmentation on new data
    print("Model updated successfully.")

def schedule_updates(interval_hours=24):
    """
    Schedule regular updates for the segmentation model.
    """
    schedule.every(interval_hours).hours.do(update_model)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    schedule_updates()
