from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import json
import seaborn as sns
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, EarlyStopping


BASE_PATH = './dataset'


def load_view_data(view_name):
    view_name = view_name.upper()

    view_base_path = f'{BASE_PATH}/{view_name}'
    print(f"Processing {view_name} data...")

    metadata = pd.read_csv(f'{view_base_path}/FileList.csv')
    volume_tracings = pd.read_csv(f'{view_base_path}/VolumeTracings.csv')

    video_features = np.load(f'{view_base_path}/video_features.npy', mmap_mode='r').astype('float16')
    ef_values = np.load(f'{view_base_path}/ef_values.npy')

    return metadata, volume_tracings, video_features, ef_values


def process_volume_tracings(volume_tracings):
    volume_features = volume_tracings.groupby('FileName').agg({
        'X': ['mean', 'std', 'min', 'max'],
        'Y': ['mean', 'std', 'min', 'max']
    }).reset_index()

    volume_features.columns = ['_'.join(col).strip() for col in volume_features.columns.values]

    return volume_features


def process_data(metadata, volume_tracings, video_features, ef_values, view_name):
    print(f"\nProcessing {view_name} data:")

    n_videos = len(video_features)
    valid_filenames = metadata.iloc[:n_videos]['FileName'].tolist()

    filtered_metadata = metadata[metadata['FileName'].isin(valid_filenames)].copy()

    volume_stats = volume_tracings[volume_tracings['FileName'].isin(valid_filenames)].groupby('FileName').agg({
        'X': ['mean', 'std', 'min', 'max'],
        'Y': ['mean', 'std', 'min', 'max']
    }).reset_index()

    volume_stats.columns = ['FileName'] + [f'{x}_{y}' for x, y in volume_stats.columns[1:]]

    final_data = pd.merge(filtered_metadata, volume_stats, on='FileName', how='inner')

    numerical_features = ['Age', 'Weight', 'Height'] + [col for col in volume_stats.columns if col != 'FileName']

    demographic_features = final_data[numerical_features].values

    volume_stats.columns = ['FileName'] + [
        f'{x}_{y}' for x, y in volume_stats.columns[1:]
        if y not in ['<lambda_0>', '<lambda_1>']
    ] + [f'{x}_q1' for x in ['X', 'Y']] + [f'{x}_q3' for x in ['X', 'Y']]

    print(f"{view_name} final shapes:")
    print(f"Video features: {video_features.shape}")
    print(f"Demographic features: {demographic_features.shape}")
    print(f"EF values: {ef_values[:n_videos].shape}")

    return demographic_features, video_features, ef_values[:n_videos]


def detailed_process_data(metadata, volume_tracings, video_features, ef_values, view_name):
    print(f"\nProcessing {view_name} data:")
    n_videos = len(video_features)
    valid_filenames = metadata.iloc[:n_videos]['FileName'].tolist()

    volume_stats = volume_tracings[volume_tracings['FileName'].isin(valid_filenames)].groupby('FileName').agg({
        'X': ['mean', 'std', 'min', 'max', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)],
        'Y': ['mean', 'std', 'min', 'max', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]
    }).reset_index()

    volume_stats.columns = ['FileName'] + [
        f'{x}_{y}' for x, y in volume_stats.columns[1:]
        if y not in ['<lambda_0>', '<lambda_1>']
    ] + [f'{x}_q1' for x in ['X', 'Y']] + [f'{x}_q3' for x in ['X', 'Y']]

    filtered_metadata = metadata[metadata['FileName'].isin(valid_filenames)].copy()

    final_data = pd.merge(filtered_metadata, volume_stats, on='FileName', how='inner')

    final_data['BMI'] = final_data['Weight'] / ((final_data['Height'] / 100) ** 2)

    final_data['Age_Category'] = pd.cut(final_data['Age'],
                                        bins=[0, 30, 45, 60, 75, 100],
                                        labels=['Young', 'Middle-Age', 'Early-Senior', 'Senior', 'Elderly']
                                        )

    final_data['X_Range'] = final_data['X_max'] - final_data['X_min']
    final_data['Y_Range'] = final_data['Y_max'] - final_data['Y_min']
    final_data['Aspect_Ratio'] = final_data['X_Range'] / final_data['Y_Range']

    numerical_features = [
        'Age', 'Weight', 'Height', 'BMI',
        'X_mean', 'X_std', 'X_min', 'X_max', 'X_median', 'X_q1', 'X_q3',
        'Y_mean', 'Y_std', 'Y_min', 'Y_max', 'Y_median', 'Y_q1', 'Y_q3',
        'X_Range', 'Y_Range', 'Aspect_Ratio'
    ]

    view_encoded = pd.DataFrame([view_name] * len(final_data), columns=['View'])
    view_encoded = pd.get_dummies(view_encoded['View'], prefix='View')

    age_encoded = pd.get_dummies(final_data['Age_Category'], prefix='Age_Group')

    demographic_features = np.hstack([
        final_data[numerical_features].values,
        age_encoded.values,
        view_encoded.values
    ])

    print(f"{view_name} final shapes:")
    print(f"Video features: {video_features.shape}")
    print(f"Demographic features: {demographic_features.shape}")
    print(f"EF values: {ef_values[:n_videos].shape}")

    return demographic_features, video_features, ef_values[:n_videos]


def detailed_process_data_iteration_2(metadata, volume_tracings, video_features, ef_values, view_name):
    print(f"\nProcessing {view_name} data:")
    n_videos = len(video_features)
    valid_filenames = metadata.iloc[:n_videos]['FileName'].tolist()

    volume_stats = volume_tracings[volume_tracings['FileName'].isin(valid_filenames)].groupby('FileName').agg({
        'X': ['mean', 'std', 'min', 'max', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)],
        'Y': ['mean', 'std', 'min', 'max', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]
    }).reset_index()

    volume_stats.columns = ['FileName'] + [
        f'{x}_{y}' for x, y in volume_stats.columns[1:]
        if y not in ['<lambda_0>', '<lambda_1>']
    ] + [f'{x}_q1' for x in ['X', 'Y']] + [f'{x}_q3' for x in ['X', 'Y']]

    filtered_metadata = metadata[metadata['FileName'].isin(valid_filenames)].copy()

    final_data = pd.merge(filtered_metadata, volume_stats, on='FileName', how='inner')

    final_data['BMI'] = final_data['Weight'] / ((final_data['Height'] / 100) ** 2)

    final_data['Age_Category'] = pd.cut(final_data['Age'],
                                        bins=[0, 30, 45, 60, 75, 100],
                                        labels=['Young', 'Middle-Age', 'Early-Senior', 'Senior', 'Elderly']
                                        )

    final_data['X_Range'] = final_data['X_max'] - final_data['X_min']
    final_data['Y_Range'] = final_data['Y_max'] - final_data['Y_min']
    final_data['Aspect_Ratio'] = final_data['X_Range'] / final_data['Y_Range']

    numerical_features = [
        'Age', 'Weight', 'Height', 'BMI',
        'X_mean', 'X_std', 'X_min', 'X_max', 'X_median', 'X_q1', 'X_q3',
        'Y_mean', 'Y_std', 'Y_min', 'Y_max', 'Y_median', 'Y_q1', 'Y_q3',
        'X_Range', 'Y_Range', 'Aspect_Ratio'
    ]

    view_encoded = pd.DataFrame([view_name] * len(final_data), columns=['View'])
    view_encoded = pd.get_dummies(view_encoded['View'], prefix='View')

    age_encoded = pd.get_dummies(final_data['Age_Category'], prefix='Age_Group')

    demographic_features = np.hstack([
        final_data[numerical_features].values,
        age_encoded.values,
        view_encoded.values
    ])

    final_data['Volume_Variability'] = final_data['X_std'] * final_data['Y_std']
    final_data['Circularity'] = (4 * np.pi * final_data['X_mean'] * final_data['Y_mean']) / \
                                (final_data['X_Range'] ** 2 + final_data['Y_Range'] ** 2)

    final_data['Heart_Rate_Efficiency'] = final_data['Weight'] / final_data['Age']

    numerical_features.extend([
        'Volume_Variability',
        'Circularity',
        'Heart_Rate_Efficiency'
    ])

    print(f"{view_name} final shapes:")
    print(f"Video features: {video_features.shape}")
    print(f"Demographic features: {demographic_features.shape}")
    print(f"EF values: {ef_values[:n_videos].shape}")

    return demographic_features, video_features, ef_values[:n_videos]


def load_and_combine_views(views=['a4c', 'psax']):
    views = [view.upper() for view in views]

    combined_video = []
    combined_demographic = []
    combined_ef = []

    for view in views:
        if view == 'A4C':
            a4c_metadata, a4c_tracings, a4c_video_features, a4c_ef_values = load_view_data('A4C')
            a4c_demographic, a4c_video, a4c_ef = process_data(
                a4c_metadata, a4c_tracings, a4c_video_features, a4c_ef_values, 'A4C'
            )

            a4c_view_indicator = np.ones((a4c_demographic.shape[0], 1))
            a4c_demographic = np.hstack([a4c_demographic, a4c_view_indicator])

            combined_video.append(a4c_video)
            combined_demographic.append(a4c_demographic)
            combined_ef.append(a4c_ef)

        elif view == 'PSAX':
            psax_metadata, psax_tracings, psax_video_features, psax_ef_values = load_view_data('PSAX')
            psax_demographic, psax_video, psax_ef = process_data(
                psax_metadata, psax_tracings, psax_video_features, psax_ef_values, 'PSAX'
            )

            psax_view_indicator = np.zeros((psax_demographic.shape[0], 1))
            psax_demographic = np.hstack([psax_demographic, psax_view_indicator])

            combined_video.append(psax_video)
            combined_demographic.append(psax_demographic)
            combined_ef.append(psax_ef)

        else:
            raise ValueError(f"Unsupported view type: {view}. Supported views are 'A4C' and 'PSAX'.")

    combined_video = np.vstack(combined_video)
    combined_demographic = np.vstack(combined_demographic)
    combined_ef = np.concatenate(combined_ef)

    print("\nCombined data shapes:")
    print(f"Video features: {combined_video.shape}")
    print(f"Demographic features: {combined_demographic.shape}")
    print(f"EF values: {combined_ef.shape}")

    return combined_video, combined_demographic, combined_ef


def create_model(video_shape, demographic_shape):
    view_input = Input(shape=video_shape)
    view_flattened = TimeDistributed(Flatten())(view_input)
    view_lstm = LSTM(256, return_sequences=True)(view_flattened)
    view_lstm = Dropout(0.5)(view_lstm)
    view_lstm = LSTM(128)(view_lstm)
    view_lstm = Dropout(0.5)(view_lstm)

    view_demographic_input = Input(shape=(demographic_shape,))
    view_demographic_dense = Dense(64, activation='relu')(view_demographic_input)

    view_combined = Concatenate()([view_lstm, view_demographic_dense])

    final_combined = Dense(128, activation='relu')(view_combined)
    final_combined = Dense(64, activation='relu')(final_combined)
    output = Dense(1, activation='linear')(final_combined)

    model = Model(inputs=[view_input, view_demographic_input], outputs=output)

    return model


class TrainingMetricsTracker:
    def __init__(self, model_name):
        self.metrics = {
            'model_metadata': {
                'model_name': model_name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'epoch_metrics': [],
            'learning_dynamics': {
                'loss_values': [],
                'mae_values': []
            },
            'performance_breakdown': {
                'per_epoch_performance': {},
                'view_specific_metrics': {}
            },
            'batch_metrics': []
        }

        self.output_dir = f'./model_tracking_{model_name}'
        os.makedirs(self.output_dir, exist_ok=True)

    def track_epoch_metrics(self, epoch, logs):
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': logs.get('loss', 0),
            'val_loss': logs.get('val_loss', 0),
            'train_mae': logs.get('mae', 0),
            'val_mae': logs.get('val_mae', 0)
        }

        self.metrics['epoch_metrics'].append(epoch_metrics)

        self.metrics['learning_dynamics']['loss_values'].append(epoch_metrics['train_loss'])
        self.metrics['learning_dynamics']['mae_values'].append(epoch_metrics['train_mae'])

    def track_view_specific_performance(self, view_name, performance_metrics):
        self.metrics['performance_breakdown']['view_specific_metrics'][view_name] = performance_metrics

    def save_metrics(self):
        metrics_path = os.path.join(self.output_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        self._plot_loss_curves()
        self._plot_learning_dynamics()
        self._plot_view_performance()

    def _plot_loss_curves(self):
        df = pd.DataFrame(self.metrics['epoch_metrics'])
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['train_mae'], label='Training MAE')
        plt.plot(df['epoch'], df['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error Curves')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_and_mae_curves.png'))
        plt.close()

    def _plot_learning_dynamics(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['learning_dynamics']['loss_values'])
        plt.title('Loss Values')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['learning_dynamics']['mae_values'])
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'learning_dynamics.png'))
        plt.close()

    def _plot_view_performance(self):
        views = list(self.metrics['performance_breakdown']['view_specific_metrics'].keys())
        mae_scores = [
            metrics.get('mae', 0) for metrics in
            self.metrics['performance_breakdown']['view_specific_metrics'].values()
        ]

        plt.figure(figsize=(8, 5))
        plt.bar(views, mae_scores)
        plt.title('Performance by View')
        plt.xlabel('View')
        plt.ylabel('Mean Absolute Error')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'view_performance.png'))
        plt.close()


def create_tracking_callback(metrics_tracker):
    class MetricsTrackerCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            metrics_tracker.track_epoch_metrics(epoch, logs or {})

        def on_train_batch_end(self, batch, logs=None):
            if logs and 'loss' in logs:
                metrics_tracker.metrics['batch_metrics'].append({
                    'batch': batch,
                    'loss': logs.get('loss', 0)
                })

    return MetricsTrackerCallback()


def train_model(views):
    video, demographic, ef = load_and_combine_views(views)

    X_video_train, X_video_val, X_demo_train, X_demo_val, y_train, y_val = train_test_split(
        video,
        demographic,
        ef,
        test_size=0.2,
        random_state=42
    )

    video_shape = video.shape[1:]
    demographic_shape = demographic.shape[1]

    model = create_model(video_shape, demographic_shape)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mean_squared_error',
        metrics=['mae']
    )

    history = model.fit(
        [X_video_train, X_demo_train],
        y_train,
        validation_data=([X_video_val, X_demo_val], y_val),
        epochs=10,
        batch_size=8
    )

    return model, history


def train_model_iteration_one(views):
    video, demographic, ef = load_and_combine_views(views)
    X_video_train, X_video_val, X_demo_train, X_demo_val, y_train, y_val = train_test_split(
        video,
        demographic,
        ef,
        test_size=0.2,
        random_state=42
    )
    video_shape = video.shape[1:]
    demographic_shape = demographic.shape[1]
    model = create_model(video_shape, demographic_shape)

    # Initialize metrics tracker
    model_name = f"EF_Predictor_{'_'.join(views)}"
    metrics_tracker = TrainingMetricsTracker(model_name)
    tracking_callback = create_tracking_callback(metrics_tracker)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mean_squared_error',
        metrics=['mae']
    )

    history = model.fit(
        [X_video_train, X_demo_train],
        y_train,
        validation_data=([X_video_val, X_demo_val], y_val),
        epochs=10,
        batch_size=8,
        callbacks=[tracking_callback]
    )

    # Track view-specific performance
    metrics_tracker.track_view_specific_performance('Combined', {
        'mae': np.mean(history.history['val_mae']),
        'loss': np.mean(history.history['val_loss'])
    })

    # Save metrics and generate visualizations
    metrics_tracker.save_metrics()

    return model, history, metrics_tracker


def augment_video_data(video_features, augmentation_factor=1.5):
    augmented_features = np.copy(video_features)

    noise = np.random.normal(
        0,
        0.01 * augmentation_factor,
        augmented_features.shape
    )
    augmented_features += noise

    dropout_mask = np.random.random(augmented_features.shape[:2]) > (0.1 * augmentation_factor)
    augmented_features *= dropout_mask[:, :, np.newaxis, np.newaxis, np.newaxis]

    return augmented_features


def train_model_iteration_2(views):
    video, demographic, ef = load_and_combine_views(views)

    video = augment_video_data(video)

    X_video_train, X_video_val, X_demo_train, X_demo_val, y_train, y_val = train_test_split(
        video,
        demographic,
        ef,
        test_size=0.2,
        random_state=42
    )

    video_shape = video.shape[1:]
    demographic_shape = demographic.shape[1]
    model = create_model(video_shape, demographic_shape)

    lr_scheduler = LearningRateScheduler(
        lambda epoch: 1e-4 * (0.5 ** np.floor((1 + epoch) / 5))
    )

    early_stopping = EarlyStopping(
        monitor='val_mae',
        patience=10,
        restore_best_weights=True
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mean_squared_error',
        metrics=['mae']
    )

    history = model.fit(
        [X_video_train, X_demo_train],
        y_train,
        validation_data=([X_video_val, X_demo_val], y_val),
        epochs=50,
        batch_size=8,
        callbacks=[
            lr_scheduler,
            early_stopping,
            tracking_callback
        ]
    )

    return model, history, metrics_tracker


def evaluate(model_path, view_to_evaluate):
    print(f"\n\nEvaluation For {view_to_evaluate} Starting..\n\n")

    metadata, tracings, video_features, ef_values = load_view_data(view_to_evaluate)

    demographic, video, ef = process_data(
        metadata, tracings, video_features, ef_values, view_to_evaluate
    )

    X_train, X_val, X_demo_train, X_demo_val, y_train, y_val = train_test_split(
        video,
        demographic,
        ef,
        test_size=0.2,
        random_state=42
    )

    loaded_model = load_model(model_path)

    print("Input data shape:", X_demo_val.shape)
    print("Model's expected input shape:", loaded_model.input_shape)

    y_pred = loaded_model.predict([X_val, X_demo_val])

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)

    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R2) Score:", r2)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel("Actual Ejection Fraction")
    plt.ylabel("Predicted Ejection Fraction")
    plt.title("Actual vs Predicted Ejection Fraction")
    plt.show()

    print(f"\n\nEvaluation For {view_to_evaluate} Ended.")


def main(view_to_train):
    model, history = train_model([view_to_train])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"{view_to_train}_model_{timestamp}.keras"
    model.save(model_filename)
