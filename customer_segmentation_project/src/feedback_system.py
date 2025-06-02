import json
from datetime import datetime

class FeedbackSystem:
    def __init__(self, feedback_file='feedback.json'):
        self.feedback_file = feedback_file
        self.feedback_data = self.load_feedback()
    
    def load_feedback(self):
        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_feedback(self):
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f)
    
    def add_feedback(self, user_id, segment_id, rating, comments):
        feedback = {
            'user_id': user_id,
            'segment_id': segment_id,
            'rating': rating,
            'comments': comments,
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_data.append(feedback)
        self.save_feedback()
    
    def get_feedback_for_segment(self, segment_id):
        return [f for f in self.feedback_data if f['segment_id'] == segment_id]
    
    def get_average_rating_for_segment(self, segment_id):
        segment_feedback = self.get_feedback_for_segment(segment_id)
        if not segment_feedback:
            return None
        return sum(f['rating'] for f in segment_feedback) / len(segment_feedback)

def analyze_feedback(feedback_system):
    """
    Analyze feedback to identify areas for model improvement.
    """
    all_segments = set(f['segment_id'] for f in feedback_system.feedback_data)
    for segment in all_segments:
        avg_rating = feedback_system.get_average_rating_for_segment(segment)
        if avg_rating is not None:
            print(f"Segment {segment} - Average Rating: {avg_rating:.2f}")
            if avg_rating < 3.5:  # Threshold for identifying segments that need improvement
                print(f"  Segment {segment} needs improvement. Analyzing feedback...")
                segment_feedback = feedback_system.get_feedback_for_segment(segment)
                # Analyze comments for common themes (in a real scenario, you might use NLP techniques here)
                common_words = [word for f in segment_feedback for word in f['comments'].lower().split() if len(word) > 3]
                print(f"  Common themes in feedback: {', '.join(set(common_words[:5]))}")

# Example usage
if __name__ == "__main__":
    feedback_system = FeedbackSystem()
    
    # Simulating user feedback
    feedback_system.add_feedback(1, 'A', 4, "Great segmentation, very accurate")
    feedback_system.add_feedback(2, 'A', 5, "Perfectly captures my buying behavior")
    feedback_system.add_feedback(3, 'B', 2, "Doesn't seem accurate, needs improvement")
    
    analyze_feedback(feedback_system)
