import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import signal

# Tudatpy imports
from tudatpy.interface import spice
from tudatpy.astro import time_representation, element_conversion
from tudatpy.astro.time_representation import DateTime
from tudatpy.dynamics import environment_setup, environment
from tudatpy.estimation import observable_models_setup, observations_setup
from tudatpy import estimation
from tudatpy.math import interpolators
import constants #

# %% 1. Configuration & Paths
fdets_folder = "/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/LEGA/"
output_folder = "/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/LEGA_shifted/"

time_shift = 0

# Find all matching files
fdets_files = glob.glob(os.path.join(fdets_folder, "Fdets.jui2024.08.20.*.r2i.txt"))

# %% 2. Helper Functions
def extract_base_frequency(file_path):
    """Parses the FDETS header to find the Base frequency in MHz and returns it in Hz."""
    with open(file_path, 'r') as f:
        for line in f:
            if "Base frequency:" in line:
                # Matches digits and decimals after "Base frequency:"
                match = re.search(r"Base frequency:\s*([\d\.]+)\s*MHz", line)
                if match:
                    return float(match.group(1)) * 1e6
    raise ValueError(f"Could not find Base frequency in {file_path}")

def strip_and_shift_first_column(input_file, output_folder, time_shift_seconds=0.0):
    """Removes the scan index and shifts the UTC time tag."""
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
                    utc_str = parts[1]
                    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in utc_str else "%Y-%m-%dT%H:%M:%S"
                    dt = datetime.strptime(utc_str, fmt)
                    dt_shifted = dt + timedelta(seconds=time_shift_seconds)
                    parts[1] = dt_shifted.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                    f_out.write(" ".join(parts[1:]) + '\n')
    return output_path

# %% 3. Main Loop per Station
for f_path in fdets_files:
    station_id = os.path.basename(f_path).split('.')[4]
    site_full_name = constants.ID_TO_SITE.get(station_id)

    if not site_full_name or site_full_name not in constants.STATION_GEODETIC_POSITIONS:
        continue

    # Dynamically extract the frequency for this specific file
    current_base_freq = extract_base_frequency(f_path)
    print(f"Processing {station_id} ({site_full_name}) at {current_base_freq/1e6:.2f} MHz...")

    # Environment Setup
    spice.load_standard_kernels()
    spice.load_kernel("juice_archive/spk/juice_orbc_000097_230414_310721_v02.bsp")

    start = datetime(2024, 8, 19); end = datetime(2024, 8, 21)
    start_time = DateTime.from_python_datetime(start).to_epoch()
    end_time = DateTime.from_python_datetime(end).to_epoch()
    start_time_buffer = start_time - 86400; end_time_buffer = end_time + 86400


    body_settings = environment_setup.get_default_body_settings_time_limited(
        ["Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"],
        start_time, end_time, "SSB", "J2000")

    # --- Add these to the Earth setup in your loop ---
    global_frame_origin = "SSB"; global_frame_orientation = "J2000"
    body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical_spice()
    body_settings.get('Earth').gravity_field_settings.associated_reference_frame = "ITRS"

    spacecraft_name = "JUICE"; spacecraft_central_body = "Jupiter"
    body_settings.add_empty_settings(spacecraft_name)
    body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
        start_time_buffer, end_time_buffer, 10.0, spacecraft_central_body, global_frame_orientation)
    body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
        global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")


    coords = constants.STATION_GEODETIC_POSITIONS[site_full_name]
    tudat_coords = [coords[0], np.deg2rad(coords[1]), np.deg2rad(coords[2])]


    body_settings.get('Earth').ground_station_settings = (
            environment_setup.ground_station.radio_telescope_stations() +
            [environment_setup.ground_station.basic_station(station_id, tudat_coords, element_conversion.geodetic_position_type)]
    )

    body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
        environment_setup.rotation_model.iau_2006, "J2000",
        interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), start_time_buffer, end_time_buffer, 3600.0),
        interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), start_time_buffer, end_time_buffer, 3600.0),
        interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), start_time_buffer, end_time_buffer, 10.0))

    body_settings.add_empty_settings("JUICE")
    body_settings.get("JUICE").ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
        start_time_buffer, end_time_buffer, 10.0, "Jupiter", "J2000")

    bodies = environment_setup.create_system_of_bodies(body_settings)

    vehicleSys = environment.VehicleSystems()
    vehicleSys.set_default_transponder_turnaround_ratio_function()
    bodies.get_body("JUICE").system_models = vehicleSys

    # Norcia Ramp
    station_ramp = environment.PiecewiseLinearFrequencyInterpolator(
        [DateTime(2024,8,19, 10,58,36).epoch(), DateTime(2024,8,20, 11,32,17).epoch()],
        [DateTime(2024,8,19, 22,10,22).epoch(), DateTime(2024,8,20, 19,28,9).epoch()],
        [0, 0], [7180.142419e6, 7180.127320e6]
    )
    bodies.get('Earth').get_ground_station('NWNORCIA').transmitting_frequency_calculator = station_ramp

    # %% 4. Observations & Residuals
    link_definition = observable_models_setup.links.LinkDefinition({
        observable_models_setup.links.receiver: observable_models_setup.links.body_reference_point_link_end_id('Earth', station_id),
        observable_models_setup.links.reflector1: observable_models_setup.links.body_origin_link_end_id('JUICE'),
        observable_models_setup.links.transmitter: observable_models_setup.links.body_reference_point_link_end_id('Earth', 'NWNORCIA'),
    })

    light_time_correction_list = [observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(["Sun"])]
    observable_models_setup.light_time_corrections.set_vmf_troposphere_data(
        ["juice_archive/vmf/2024231.vmf3_g", "juice_archive/vmf/2024232.vmf3_g",
         "juice_archive/vmf/2024233.vmf3_g", "juice_archive/vmf/2024234.vmf3_g"],
        True, False, bodies, True, True)

    fdets_clean = strip_and_shift_first_column(f_path, output_folder, time_shift)

    station_positions = {
        station_id: bodies.get('Earth').get_ground_station(station_id).station_state.get_cartesian_position(start_time),
        'NWNORCIA': bodies.get('Earth').get_ground_station('NWNORCIA').station_state.get_cartesian_position(start_time)
    }

    fdets_collection = observations_setup.observations_wrapper.observations_from_fdets_files(
        fdets_clean, current_base_freq,
        ["utc_datetime_string", "signal_to_noise_ratio", "normalised_spectral_max", "doppler_measured_frequency_hz", "doppler_noise_hz"],
        "JUICE", 'NWNORCIA', station_id,
        observations_setup.ancillary_settings.FrequencyBands.x_band,
        observations_setup.ancillary_settings.FrequencyBands.x_band, station_positions
    )

    simulators = observations_setup.observations_simulation_settings.create_observation_simulators(
        [observable_models_setup.model_settings.doppler_measured_frequency(link_definition, light_time_correction_list)], bodies)
    estimation.observations.compute_residuals_and_dependent_variables(fdets_collection, simulators, bodies)

    # %% 5. Visualization
    times_tdb = fdets_collection.get_concatenated_observation_times()
    residuals = fdets_collection.get_concatenated_residuals()
    residuals_detrended = signal.detrend(residuals)

    converter = time_representation.default_time_scale_converter()
    times_datetime = np.array([DateTime.to_python_datetime(DateTime.from_epoch(
        converter.convert_time(time_representation.tdb_scale, time_representation.utc_scale, t))) for t in times_tdb])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(times_datetime, residuals, color='gray', s=1, alpha=0.3, label='Original')
    ax.scatter(times_datetime, residuals_detrended, color='green', s=2, label='Detrended')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title(f'JUICE Residuals: {station_id} ({site_full_name}) @ {current_base_freq/1e6:.2f} MHz')
    ax.set_ylabel('Residual [Hz]'); ax.set_xlabel('Time (UTC)')
    ax.legend(title=f"Original RMS: {np.sqrt(np.mean(residuals**2)):.4f} Hz\nDetrended RMS: {np.sqrt(np.mean(residuals_detrended**2)):.4f} Hz")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()