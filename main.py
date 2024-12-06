from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
