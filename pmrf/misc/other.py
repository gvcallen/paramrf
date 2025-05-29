from datetime import datetime

def time_string(format="%H:%M:%S"):
    return datetime.now().strftime(format)