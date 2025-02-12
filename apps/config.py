from keras import metrics


class LSTMConfig:
    epochs = 1
    batch_size = 32
    validation_split = 0.2
    learning_rate = 0.1
    metrics = ['accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC()]
    test_size = 0.2
    time_step = 60
    layers = 2
    neurons = 20
    dropout = 0.2
    dense_units = 25
