from evaluate import load


def compute_metrics(predictions, label_ids):
    metric = load('squad')
    return metric.compute(predictions=predictions, references=label_ids)
