import traceback
import sys

try:
    from model import EmotionDetectionModel
    model = EmotionDetectionModel()
    print("SUCCESS: Model loaded!", flush=True)
except Exception as e:
    print("ERROR:", str(e), flush=True)
    traceback.print_exc(file=sys.stdout)