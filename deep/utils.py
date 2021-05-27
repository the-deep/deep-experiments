from datetime import datetime


def formatted_time():
    time = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    milliseconds = datetime.now().microsecond // 1000
    return f"{time}-{milliseconds:03d}"
