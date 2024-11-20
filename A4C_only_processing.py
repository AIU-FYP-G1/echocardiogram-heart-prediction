import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Flatten, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

A4C_PATH = './dataset/A4C'
PSAX_PATH = './dataset/PSAX'


def load_view_data(base_path, view_name):
    print(f"Processing {view_name} data...")

    metadata = pd.read_csv(f'{base_path}/FileList.csv')
    volume_tracings = pd.read_csv(f'{base_path}/VolumeTracings.csv')

    video_features = np.load(f'{base_path}/video_features.npy', mmap_mode='r')
    ef_values = np.load(f'{base_path}/ef_values.npy')

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

    print(f"{view_name} final shapes:")
    print(f"Video features: {video_features.shape}")
    print(f"Demographic features: {demographic_features.shape}")
    print(f"EF values: {ef_values[:n_videos].shape}")

    return demographic_features, video_features, ef_values[:n_videos]


def create_model(video_shape, demographic_shape):
    a4c_input = Input(shape=video_shape)
    a4c_flattened = TimeDistributed(Flatten())(a4c_input)
    a4c_lstm = LSTM(256, return_sequences=True)(a4c_flattened)
    a4c_lstm = Dropout(0.5)(a4c_lstm)
    a4c_lstm = LSTM(128)(a4c_lstm)
    a4c_lstm = Dropout(0.5)(a4c_lstm)

    a4c_demographic_input = Input(shape=(demographic_shape,))
    a4c_demographic_dense = Dense(64, activation='relu')(a4c_demographic_input)

    a4c_combined = Concatenate()([a4c_lstm, a4c_demographic_dense])

    final_combined = Dense(128, activation='relu')(a4c_combined)
    final_combined = Dense(64, activation='relu')(final_combined)
    output = Dense(1, activation='linear')(final_combined)

    model = Model(inputs=[a4c_input, a4c_demographic_input], outputs=output)

    return model


def main():
    a4c_metadata, a4c_tracings, a4c_video_features, a4c_ef_values = load_view_data(A4C_PATH, 'A4C')

    a4c_demographic, a4c_video, a4c_ef = process_data(
        a4c_metadata, a4c_tracings, a4c_video_features, a4c_ef_values, 'A4C'
    )

    X_a4c_train, X_a4c_val, X_a4c_demo_train, X_a4c_demo_val, y_train, y_val = train_test_split(
        a4c_video,
        a4c_demographic,
        a4c_ef,
        test_size=0.2,
        random_state=42
    )

    video_shape = a4c_video.shape[1:]
    demographic_shape = a4c_demographic.shape[1]

    model = create_model(video_shape, demographic_shape)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mean_squared_error',
        metrics=['mae']
    )

    history = model.fit(
        [X_a4c_train, X_a4c_demo_train],
        y_train,
        validation_data=([X_a4c_val, X_a4c_demo_val], y_val),
        epochs=10,
        batch_size=8
    )

    loss, mae = model.evaluate([X_a4c_val, X_a4c_demo_val], y_val)
    print(f'Validation Mean Absolute Error for A4C: {mae:.2f}')

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"a4c_model_{timestamp}.keras"
    model.save(model_filename)

    return model, history
