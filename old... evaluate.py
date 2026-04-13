from utils.metrics import compute_metrics

def evaluate(model, data):
    pred = model(data)
    return compute_metrics(data.y, pred)
