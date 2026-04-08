# %%
from tudatpy.interface import spice
from tudatpy.astro import time_representation, element_conversion
from tudatpy.astro.time_representation import DateTime
from datetime import datetime
from tudatpy.dynamics import environment_setup, environment
from tudatpy.estimation import observable_models_setup, observations_setup
from tudatpy import estimation

import os
from tudatpy.math import interpolators
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tudatpy.data as data

# %% [markdown]
# ## 2. Helper Functions
# (Kept as requested)

# %%
def make_state_interpolator(times: np.ndarray, states: np.ndarray):
    interpolators = [interp1d(times, states[:, i], kind='linear', fill_value="extrapolate") for i in range(6)]
    def state_function(t: float) -> np.ndarray:
        return np.array([[f(t)] for f in interpolators])
    return state_function

def compute_scipy_quadrature(interpolated_function, times, integration_time=10):
    results = list()
    midpoints = list()
    a = min(times); max_time = max(times)
    while a + integration_time <= max_time:
        b = a + integration_time
        midpoint = (a + b) / 2
        result, _ = quad(interpolated_function, a, b)
        normalized_result = result / (b - a)
        results.append(normalized_result)
        midpoints.append(midpoint); a = b
    return results, midpoints

def strip_first_column(input_file, output_folder):
    """Utility to align FDETS data with Tudat's parser columns."""
    filename = os.path.basename(input_file)
    output_path = os.path.join(output_folder, filename)
    os.makedirs(output_folder, exist_ok=True)
    with open(input_file, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('#') or not line.strip():
                f_out.write(line)
            else:
                parts = line.split()
                if len(parts) >= 6:
                    f_out.write(" ".join(parts[1:]) + '\n')
    return output_path

# %% [markdown]
# ## 3. Simulation Setup

# %%
# Load Required Spice Kernels
spice.load_standard_kernels()
spice.load_kernel("juice_archive/spk/juice_orbc_000097_230414_310721_v02.bsp")

# Define Simulation Dates
start = datetime(2024, 8, 19)
end = datetime(2024, 8, 21)
start_time = DateTime.from_python_datetime(start).to_epoch()
end_time = DateTime.from_python_datetime(end).to_epoch()
start_time_buffer = start_time - 86400
end_time_buffer = end_time + 86400

# Create Environment
bodies_to_create = ["Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"]
global_frame_origin = "SSB"; global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings_time_limited(
    bodies_to_create, start_time, end_time, global_frame_origin, global_frame_orientation)

# Modify Earth settings
body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical_spice()
body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
    environment_setup.rotation_model.iau_2006, global_frame_orientation,
    interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), start_time_buffer, end_time_buffer, 3600.0),
    interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), start_time_buffer, end_time_buffer, 3600.0),
    interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), start_time_buffer, end_time_buffer, 10.0))
body_settings.get('Earth').gravity_field_settings.associated_reference_frame = "ITRS"

# Setup JUICE
spacecraft_name = "JUICE"
body_settings.add_empty_settings(spacecraft_name)
body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
    start_time_buffer, end_time_buffer, 10.0, "Jupiter", global_frame_orientation)
body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
    global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")

# Add Ground Stations (Naming 'Yg' to match FDETS files)
stations_to_add = {
    'Yg': [250.0, np.deg2rad(-29.0464), np.deg2rad(115.3456)],
    'NWNORCIA': [470.0, np.deg2rad(-31.0482), np.deg2rad(116.1928)]
}
new_gs_settings = []
for name, pos in stations_to_add.items():
    new_gs_settings.append(environment_setup.ground_station.basic_station(name, pos, element_conversion.geodetic_position_type))

body_settings.get('Earth').ground_station_settings = new_gs_settings
bodies = environment_setup.create_system_of_bodies(body_settings)

# Set Transponder
vehicleSys = environment.VehicleSystems()
vehicleSys.set_default_transponder_turnaround_ratio_function()
bodies.get_body("JUICE").system_models = vehicleSys

# Set Uplink Ramping (Fixing 4-digit year to 2024)
base_frequency = 8422.49e6
transmitting_station_name = 'NWNORCIA'
ramp_times = [DateTime(2024,8,19, 10,58,36).epoch(), DateTime(2024,8,20, 11,32,17).epoch()]
ramp_ends = [DateTime(2024,8,19, 22,10,22).epoch(), DateTime(2024,8,20, 19,28,9).epoch()]
station_ramp = environment.PiecewiseLinearFrequencyInterpolator(ramp_times, ramp_ends, [0, 0], [7180.142419e6, 7180.127320e6])
bodies.get('Earth').get_ground_station(transmitting_station_name).transmitting_frequency_calculator = station_ramp

# %% [markdown]
# ## 4. Load FDETS Data and Compute Residuals

# %%
# Define involved link ends (Using 'reflector1' to match PRIDE parser)
link_ends = {
    observable_models_setup.links.receiver: observable_models_setup.links.body_reference_point_link_end_id('Earth', 'Yg'),
    observable_models_setup.links.reflector1: observable_models_setup.links.body_origin_link_end_id('JUICE'),
    observable_models_setup.links.transmitter: observable_models_setup.links.body_reference_point_link_end_id('Earth', 'NWNORCIA'),
}
link_definition = observable_models_setup.links.LinkDefinition(link_ends)

# Define Model with Troposphere and Relativistic corrections
light_time_correction_list = [
    observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(["Sun"]),
    observable_models_setup.light_time_corrections.saastamoinen_tropospheric_light_time_correction()
]
open_loop_observation_model_settings = [
    observable_models_setup.model_settings.doppler_measured_frequency(link_definition, light_time_correction_list)
]
observation_simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
    open_loop_observation_model_settings, bodies)

# Load FDETS File
fdets_original = "/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/LEGA/Fdets.jui2024.08.20.Yg.r2i.txt"
fdets_clean = strip_first_column(fdets_original, "/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/LEGA_stripped/")

column_types = ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max", "doppler_measured_frequency_hz", "doppler_noise_hz"]
station_positions = {'Yg': bodies.get_body("Earth").get_ground_station('Yg').station_state.get_cartesian_position(0.0)}

print("Loading FDETS observations...")
fdets_collection = observations_setup.observations_wrapper.observations_from_fdets_files(
    fdets_clean, base_frequency, column_types, "JUICE", 'NWNORCIA', 'Yg',
    observations_setup.ancillary_settings.FrequencyBands.x_band, observations_setup.ancillary_settings.FrequencyBands.x_band,
    station_positions
)

# Compute Residuals
print("Computing residuals (O-C)...")
estimation.observations.compute_residuals_and_dependent_variables(fdets_collection, observation_simulators, bodies)

# Extract Data for Plotting
times_tdb = fdets_collection.get_concatenated_observation_times()
observed_val = fdets_collection.get_concatenated_observations()
residuals = fdets_collection.get_concatenated_residuals()
simulated_val = observed_val - residuals

# Explicit TDB to UTC conversion for plotting to avoid time bias
converter = time_representation.default_time_scale_converter()
times_utc_epochs = [converter.convert_time(t, time_representation.tdb_scale, time_representation.utc_scale) for t in times_tdb]
times_datetime = [DateTime.to_python_datetime(DateTime.from_epoch(t_utc)) for t_utc in times_utc_epochs]

# %% [markdown]
# ## 5. Visualization

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top: Frequencies
ax1.plot(times_datetime, observed_val, label='Observed (FDETS)', color='blue', alpha=0.6)
ax1.plot(times_datetime, simulated_val, label='Simulated (Tudat)', color='red', linestyle='--')
ax1.set_ylabel('Frequency [Hz]')
ax1.set_title('JUICE Doppler Comparison - Yarragadee')
ax1.legend(); ax1.grid(True, alpha=0.3)

# Bottom: Residuals
ax2.scatter(times_datetime, residuals, color='green', s=2, label='Residuals (O-C)')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_ylabel('Residual [Hz]')
ax2.set_xlabel('Time (UTC)')
ax2.legend(title=f'RMS: {np.sqrt(np.mean(residuals**2)):.6f} Hz')
ax2.grid(True, alpha=0.3)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()