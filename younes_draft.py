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

    video_features = np.load(f'{base_path}/video_features.npy')
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

    # First, get the filenames of videos we have features for
    n_videos = len(video_features)
    valid_filenames = metadata.iloc[:n_videos]['FileName'].tolist()

    # Filter metadata to only include these files
    filtered_metadata = metadata[metadata['FileName'].isin(valid_filenames)].copy()

    # Process volume tracings
    volume_stats = volume_tracings[volume_tracings['FileName'].isin(valid_filenames)].groupby('FileName').agg({
        'X': ['mean', 'std', 'min', 'max'],
        'Y': ['mean', 'std', 'min', 'max']
    }).reset_index()

    # Flatten multi-level columns
    volume_stats.columns = ['FileName'] + [f'{x}_{y}' for x, y in volume_stats.columns[1:]]

    # Merge metadata with volume statistics
    final_data = pd.merge(filtered_metadata, volume_stats, on='FileName', how='inner')

    # Extract numerical features
    numerical_features = ['Age', 'Weight', 'Height'] + [col for col in volume_stats.columns if col != 'FileName']

    # Create demographic features array
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

    psax_input = Input(shape=video_shape)
    psax_flattened = TimeDistributed(Flatten())(psax_input)
    psax_lstm = LSTM(256, return_sequences=True)(psax_flattened)
    psax_lstm = Dropout(0.5)(psax_lstm)
    psax_lstm = LSTM(128)(psax_lstm)
    psax_lstm = Dropout(0.5)(psax_lstm)

    a4c_demographic_input = Input(shape=(demographic_shape,))
    psax_demographic_input = Input(shape=(demographic_shape,))

    a4c_demographic_dense = Dense(64, activation='relu')(a4c_demographic_input)
    psax_demographic_dense = Dense(64, activation='relu')(psax_demographic_input)

    a4c_combined = Concatenate()([a4c_lstm, a4c_demographic_dense])
    psax_combined = Concatenate()([psax_lstm, psax_demographic_dense])

    final_combined = Concatenate()([a4c_combined, psax_combined])
    final_combined = Dense(128, activation='relu')(final_combined)
    final_combined = Dense(64, activation='relu')(final_combined)
    output = Dense(1, activation='linear')(final_combined)

    model = Model(
        inputs=[a4c_input, psax_input, a4c_demographic_input, psax_demographic_input],
        outputs=output
    )

    return model


def align_datasets(a4c_metadata, a4c_tracings, a4c_video_features, a4c_ef_values,
                   psax_metadata, psax_tracings, psax_video_features, psax_ef_values):
    """Align A4C and PSAX datasets based on common filenames."""
    # Get common filenames between A4C and PSAX
    a4c_files = a4c_metadata['FileName'].tolist()[:len(a4c_video_features)]
    psax_files = psax_metadata['FileName'].tolist()[:len(psax_video_features)]
    common_files = list(set(a4c_files) & set(psax_files))

    print("common files are:", common_files)

    # Get indices for common files in each dataset
    a4c_indices = [a4c_files.index(f) for f in common_files]
    psax_indices = [psax_files.index(f) for f in common_files]

    # Filter all data using these indices
    a4c_metadata_aligned = a4c_metadata.iloc[a4c_indices]
    a4c_video_aligned = a4c_video_features[a4c_indices]
    a4c_ef_aligned = a4c_ef_values[a4c_indices]

    psax_metadata_aligned = psax_metadata.iloc[psax_indices]
    psax_video_aligned = psax_video_features[psax_indices]
    psax_ef_aligned = psax_ef_values[psax_indices]

    # Filter tracings
    a4c_tracings_aligned = a4c_tracings[a4c_tracings['FileName'].isin(common_files)]
    psax_tracings_aligned = psax_tracings[psax_tracings['FileName'].isin(common_files)]

    return (a4c_metadata_aligned, a4c_tracings_aligned, a4c_video_aligned, a4c_ef_aligned,
            psax_metadata_aligned, psax_tracings_aligned, psax_video_aligned, psax_ef_aligned)


def main():
    a4c_metadata, a4c_tracings, a4c_video_features, a4c_ef_values = load_view_data(A4C_PATH, 'A4C')
    psax_metadata, psax_tracings, psax_video_features, psax_ef_values = load_view_data(PSAX_PATH, 'PSAX')

    # Process each view
    a4c_demographic, a4c_video, a4c_ef = process_data(
        a4c_metadata, a4c_tracings, a4c_video_features, a4c_ef_values, 'A4C'
    )

    psax_demographic, psax_video, psax_ef = process_data(
        psax_metadata, psax_tracings, psax_video_features, psax_ef_values, 'PSAX'
    )

    # Now all your data should be aligned
    # Continue with your existing code for splitting and model creation...

    # Split data
    X_a4c_train, X_a4c_val, X_psax_train, X_psax_val, \
        X_a4c_demo_train, X_a4c_demo_val, X_psax_demo_train, X_psax_demo_val, \
        y_train, y_val = train_test_split(
        a4c_video,
        psax_video,
        a4c_demographic,
        psax_demographic,
        a4c_ef,  # they should all be the same length now
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
        [X_a4c_train, X_psax_train, X_a4c_demo_train, X_psax_demo_train],
        y_train,
        validation_data=(
            [X_a4c_val, X_psax_val, X_a4c_demo_val, X_psax_demo_val],
            y_val
        ),
        epochs=10,
        batch_size=8
    )

    loss, mae = model.evaluate(
        [X_a4c_val, X_psax_val, X_a4c_demo_val, X_psax_demo_val],
        y_val
    )
    print(f'Validation Mean Absolute Error: {mae:.2f}')

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"combined_model_{timestamp}.keras"
    model.save(model_filename)

if __name__ == "__main__":
    model, history = main()