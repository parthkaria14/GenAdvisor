def one_step_ahead(series, model_func, window=50):
    preds = []
    for t in range(window, len(series)):
        train = series[t-window:t]
        model = model_func(train)
        pred = model.forecast(1)[0]
        preds.append(pred)
    return preds