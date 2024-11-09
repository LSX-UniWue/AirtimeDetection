from collections import defaultdict
from json import load as jload
from typing import Dict, Iterable

import numpy as np
import tensorflow as tf
import wandb
from numpy import load, vstack, ndarray, diff, argwhere
from numpy.lib.stride_tricks import sliding_window_view
from ruptures.metrics.hausdorff import hausdorff
from ruptures.metrics.precisionrecall import precision_recall
from ruptures.metrics.randindex import randindex
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, MaxPool1D, Dropout
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.optimizers import Adam

BATCH_SIZE = 512


# Define necessary functions
def get_metrics(y_pred: ndarray, y_true: ndarray, threshold: float = 0.5) -> Dict[str, float]:
    # y_pred = bs x window_size x 1
    # y_true = bs x window_size x 1
    result = defaultdict(float)
    _y_pred, _y_true = (y_pred >= threshold).squeeze(), y_true.astype('bool').squeeze()
    _y_pred, _y_true = diff(_y_pred, axis=-1), diff(_y_true, axis=-1)
    # _y_pred = bs x window_size - 1
    # _y_true = bs x window_size - 1
    count = 0
    for row_y_pred, row_y_true in zip(_y_pred, _y_true):
        cpds_pred, cpds_true = argwhere(row_y_pred).flatten().tolist(), argwhere(row_y_true).flatten().tolist()
        count += 1
        # Add dummy for implementation
        cpds_pred.append(row_y_true.shape[-1])
        cpds_true.append(row_y_true.shape[-1])
        if len(cpds_pred) < 2:
            cpds_pred.insert(0, 0)
        if len(cpds_true) < 2:
            cpds_true.insert(0, 0)
        result['rand_index'] += randindex(cpds_true, cpds_pred)
        result['hausdorff'] += hausdorff(cpds_true, cpds_pred)
        prec, rec = precision_recall(cpds_true, cpds_pred)
        result['recall'] += rec
        result['precision'] += prec

    return {k: v / count for k, v in result.items()}

def create_model(timesteps, num_features, un, dropout_rate, lr):
    def double_conv_block(x, n_filters):
        # Conv2D then ReLU activation
        x = Conv1D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        # Conv2D then ReLU activation
        x = Conv1D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        return x

    def downsample_block(x, n_filters):
        f = double_conv_block(x, n_filters)
        p = MaxPool1D(2)(f)
        p = Dropout(dropout_rate)(p)
        return f, p

    def upsample_block(x, conv_features, n_filters):
        # upsample
        x = Conv1DTranspose(n_filters, 3, 2, padding="same")(x)
        # concatenate
        x = tf.keras.layers.concatenate([x, conv_features])
        # dropout
        x = Dropout(dropout_rate)(x)
        # Conv2D twice with ReLU activation
        x = double_conv_block(x, n_filters)
        return x

    inputs = Input(shape=(timesteps, num_features))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, un)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 2 * un)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 4 * un)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 8 * un)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 16 * un)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 8 * un)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 4 * un)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 2 * un)
    # 9 - upsample
    u9 = upsample_block(u8, f1, un)
    # outputs
    outputs = Conv1D(1, 1, padding="same", activation="sigmoid")(u9)
    # unet model with Keras Functional API
    model = tf.keras.Model(inputs, outputs, name="U-Net")

    # You may need to adjust loss and metrics as per your needs
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           BinaryIoU(threshold=0.5, target_class_ids=[1])
                           ]
                  )

    return model


def train():
    # Initialize a new wandb run
    run = wandb.init()

    # Fetch hyperparameters for this run
    config = wandb.config
    lr = config.learning_rate
    batch_size = config.batch_size
    dropout_rate = config.dropout
    un = config.units
    ws = config.window_size
    w_step = config.window_steps
    red = config.data_reduction
    experiment_data = config.data_config

    # indices

    # Load and prepare data
    # Result should be your loaded data as a dictionary with the following entries, each a numpy array.
    # data_*: array of your sensor data in shape L x channels
    # labels_*: array of your labels in dense 0/1 representation
    result = {‘data_train’: [], ‘labels_train’: [], ‘data_validation’: [], ‘labels_validation’: [], ‘data_test’: [], ‘labels_test’: []}

    # Split data into training, validation, and testing sets
    train_data, train_labels = result['data_train'], result['labels_train']
    val_data, val_labels = result['data_validation'], result['labels_validation']
    test_data, test_labels = result['data_test'], result['labels_test']

    # Data reduction if applicable
    np.random.seed(42)  # For reproducibility
    tf.random.set_seed(42)
    if red <= 1:
        num_samples = int(len(train_data) * red)
        random_indices = np.random.choice(len(train_data), num_samples, replace=False)
        train_data = train_data[random_indices]
        train_labels = train_labels[random_indices]

    # Create the model
    timesteps, num_features = train_data.shape[1], train_data.shape[2]
    tf.keras.backend.clear_session()
    with tf.distribute.MirroredStrategy().scope():
        model = create_model(timesteps, num_features, un, dropout_rate, lr)
        model.summary()
        model.fit(train_data, train_labels, epochs=100, batch_size=batch_size,
                  validation_data=(val_data, val_labels),
                  callbacks=[EarlyStopping(monitor='val_binary_io_u', patience=5, mode='max',
                                           restore_best_weights=True)])
        model.save(f"model_{wandb.run.name}.keras")

        # Evaluate the model
        y_test = model.predict(test_data, batch_size=BATCH_SIZE)
        metrics_test = model.evaluate(x=test_data, y=test_labels,
                                      batch_size=BATCH_SIZE, return_dict=True)
        y_val = model.predict(val_data, batch_size=BATCH_SIZE)
        metrics_val = model.evaluate(x=val_data, y=val_labels,
                                     batch_size=BATCH_SIZE, return_dict=True)

    metrics_test |= get_metrics(y_pred=y_test, y_true=test_labels)
    metrics_val |= get_metrics(y_pred=y_val, y_true=val_labels)

    # Log results to wandb
    wandb.log({
        **{f'val_{k}': v for k, v in metrics_val.items()},
        **metrics_test
    })
    run.finish()


# Global setup configurations
sweep_configuration = {
    'name': 'setup3.1.0',
    'method': 'bayes',  # or 'grid', 'bayes', 'random'
    'metric': {
        'name': 'val_hausdorff',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'min': 1e-6,
            'max': 1e-2,
            'distribution': 'log_uniform_values'
        },
        'batch_size': {
            'values': [3 * BATCH_SIZE]
        },
        'dropout': {
            'min': 0.0,
            'max': 0.5,
            'distribution': 'uniform'
        },
        'units': {
            'values': [4, 8, 16, 32, 64]  # Powers of 2 from 4 to 128
        },
        'window_size': {
            'values': [400]
        },
        'window_steps': {
            'values': [5]
        },
        'data_reduction': {
            'values': [1.0]
        },
        'data_config': {
            'values': ['combined']
        }
    }
}


# Main function to run the training
def main():
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=“AirtimeDetection")
    wandb.agent(sweep_id, function=train, count=1)


if __name__ == '__main__':
    main()
