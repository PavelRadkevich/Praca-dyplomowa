from keras import metrics


class LSTMConfig:
    epochs = 1
    batch_size = 32
    validation_split = 0.2
    optimizer = 'adam'
    metrics = ['accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC()]
    test_size = 0.2
    time_step = 60
