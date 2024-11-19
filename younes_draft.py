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


def main():
    a4c_metadata, a4c_tracings, a4c_video_features, a4c_ef_values = load_view_data(A4C_PATH, 'A4C')
    psax_metadata, psax_tracings, psax_video_features, psax_ef_values = load_view_data(PSAX_PATH, 'PSAX')

    a4c_volume_features = process_volume_tracings(a4c_tracings)
    psax_volume_features = process_volume_tracings(psax_tracings)

    numerical_features = ['Age', 'Weight', 'Height']
    scaler = StandardScaler()

    a4c_metadata[numerical_features] = scaler.fit_transform(a4c_metadata[numerical_features])
    psax_metadata[numerical_features] = scaler.transform(psax_metadata[numerical_features])

    a4c_data = pd.merge(a4c_metadata, a4c_volume_features,
                        left_on='FileName', right_on='FileName_',
                        how='inner')
    psax_data = pd.merge(psax_metadata, psax_volume_features,
                         left_on='FileName', right_on='FileName_',
                         how='inner')

    common_patients = pd.merge(a4c_data, psax_data,
                               on=['PatientID'],  # Adjust based the unique identifier
                               how='inner',
                               suffixes=('_a4c', '_psax'))

    common_patients = common_patients.sample(frac=0.01, random_state=42).reset_index(drop=True)

    feature_columns = numerical_features + [col for col in a4c_volume_features.columns[1:]]
    a4c_demographic_features = common_patients[[col + '_a4c' for col in feature_columns]].values
    psax_demographic_features = common_patients[[col + '_psax' for col in feature_columns]].values

    print("Shapes:\n")
    print(f"A4C video features: {a4c_video_features.shape}")
    print(f"PSAX video features: {psax_video_features.shape}")
    print(f"A4C demographic features: {a4c_demographic_features.shape}")
    print(f"PSAX demographic features: {psax_demographic_features.shape}")
    print(f"EF values: {a4c_ef_values.shape}")

    X_a4c_train, X_a4c_val, X_psax_train, X_psax_val, \
        X_a4c_demo_train, X_a4c_demo_val, X_psax_demo_train, X_psax_demo_val, \
        y_train, y_val = train_test_split(
        a4c_video_features,
        psax_video_features,
        a4c_demographic_features,
        psax_demographic_features,
        a4c_ef_values,
        test_size=0.2,
        random_state=42
    )

    video_shape = a4c_video_features.shape[1:]
    demographic_shape = a4c_demographic_features.shape[1]

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

    return model, history


if __name__ == "__main__":
    model, history = main()