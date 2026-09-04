# %% [markdown]
# # JUICE Doppler Residuals During PRIDE Lunar-Earth Gravity Assist (LEGA)
#
# This notebook is a teaching companion to `lega_simulations_all_stations.py`.
#
# ## Requirements:
# - This was ran using a recent Tudat version, which you can build from this branch: https://github.com/tudat-team/tudatpy/tree/feature/data-refactor
# - if you have an older version of Tudat, the script will throw an Error due to a refactor of the `observation_wrapper` module. In that case, changing this line: `fdets_collection = observations.observations_from_fdets_files` to this: `fdets_collection = observatinos_setup.observations_wrapper.observations_from_fdets_files` should suffice. Otherwise, you can contact the Tudat Team for clarifications. 
#
# ## The scientific question
#
# During the LEGA, five PRIDE VLBI ground stations (Hartebeesthoek "Hh",
# Ceduna "Cd", Yarragadee "Yg", Hobart12 "Hb", Katherine12 "Ke") recorded
# the Doppler-shifted downlink signal from JUICE, while ESA's New Norcia
# ("NWNORCIA") station uplinked the reference carrier. This forms a
# **three-way Doppler link**: uplink station → spacecraft (transponds the
# signal) → downlink station.
#
# Each station's raw measurements are stored in an **FDETS file**: one row
# per detection, giving a UTC timestamp and a measured Doppler frequency.
# By simulating the same three-way link with tudatpy (using JUICE's SPICE
# trajectory kernel and the known station positions) we get a *predicted*
# Doppler frequency to compare against. The difference,
#
# $$\text{residual}(t) = f_{\text{observed}}(t) - f_{\text{simulated}}(t)$$
#
# is the classic **observed-minus-computed (O-C) residual** used throughout
# orbit determination: if the trajectory, station positions, and link model
# are all correct, the residual should be small and structureless (just
# measurement noise). Anything else (a slope, an offset, a curve) is
# telling you *something* is imperfect, and figuring out what is the point
# of this exercise (disclaimer: this is not easy to do).
#
# ## What this notebook produces
#
# For every station, for every usable pass of data, four diagnostic plots:
#
# 1. **Original, no tropospheric correction**, the raw O-C residual.
# 2. **Original, with VMF3 tropospheric correction**, same, but with an
#    atmospheric delay model applied to the light-time computation.
# 3. **Original vs Detrended (no VMF)**, the raw residual with its
#    best-fit straight line removed, to separate a *slowly varying trend*
#    (offset + drift) from *short-timescale scatter*.
# 4. **Original vs Detrended (with VMF)**, same, with VMF3 applied.
#
# Comparing these across stations is where the interesting questions live:
# some stations detrend down to a clean noise floor, others don't, and the
# *reason why* differs from station to station (possibilities include: imperfect station calibration, time-offset biases, ephemeris limitations, etc...). 
#
# This notebook gives you the tools to investigate; it deliberately does not hand you the
# answer for every station.

# %% [markdown]
# ## 1. Configuration & Paths
#
# Everything station- and time-specific lives here, so the rest of the
# notebook can stay generic.
#
# A few concepts worth understanding before reading the code:
#
# - **Why Aug 19 and Aug 20 are two separate files.** JUICE's gravity
#   assist involves two distinct close approaches on consecutive days: the
#   Moon on 2024-08-19, then Earth on 2024-08-20. Each ground station
#   recorded a separate FDETS file per day.
# - **Why there's a lunar-occultation gap.** During the lunar flyby, the
#   Moon physically blocks the line of sight between NWNORCIA's uplink and
#   JUICE for a while, forcing the uplink from two-way into one-way mode.
#   This script's three-way link model isn't valid during that gap, so it's
#   excluded from every station's `Pre_Lunar_Occultation` pass.
# - **Why Hh alone gets a "scan jump" split.** Inspecting Hh's Aug-19 FDETS
#   file shows a sudden residual discontinuity that lines up *exactly* with
#   a scan-index boundary in the raw file (**we regard this as a hint that something might be gone wrong in the processing of the observations**). Splitting Hh's lunar-flyby pass at that boundary keeps each
#   half internally consistent. No such boundary has been established for
#   the other stations, so their lunar-flyby data is kept as one
#   unsegmented pass. **Please, do not assume they don't *have* a similar artifact**,
#   just that we haven't looked for one yet (the scan-boundary markers in
#   every plot make this easy to check for any station anyway).
# - **`TIME_SHIFT_BY_STATION_S`.** A constant timing bias in a station's
#   FDETS timestamps looks, in the residual, like a straight-line offset
#   *and* slope simultaneously (both driven by the same one number). See
#   the comment in the code cell below for how that was diagnosed for
#   Yarragadee specifically. This dictionary is where you'd apply a per-station `time_shift`, if you wanna play around and see if that improves the residuals. 

# %%
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import signal
from tudatpy.data_input.environment_data import spice
from tudatpy.astro import time_representation, element_conversion
from tudatpy.astro.time_representation import DateTime
from tudatpy.dynamics import environment_setup, environment
from tudatpy.estimation import observable_models_setup, observations_setup, observations
from tudatpy import estimation
from tudatpy.math import interpolators
from tudatpy.data_input.tracking_data.fdets import FdetDateFormat
import constants

# %%
fdets_folder = "/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/LEGA/ORIGINAL_FDETS"
output_folder = "/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/LEGA/SHIFTED_FDETS/"
plots_folder = "/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/LEGA/PLOTS/"
kernels_folder = "/Users/lgisolfi/Desktop/PRIDE_DATA_NEW/LEGA/JUICE_ARCHIVE/"

# Every station whose FDETS files we have. Hh gets special treatment (see
# build_passes_for_station below); the others are each processed as a single
# pass covering all of their available observations.
STATION_IDS = ["Yg", "Hb", "Hh", "Cd", "Ke"]

# Per-station constant time-offset (seconds) applied to that station's FDETS
# timestamps before computing residuals. Defaults to 0.0 for everyone, but users
# can tweak individual entries to test/apply a station-specific timing bias.
#
# Yg is the only station where a single constant offset (0.84 s) collapses
# almost the entire raw residual down to the same level reached by linear
# detrending (RMS drops ~1.58 Hz -> ~0.008 Hz, matching the ~0.006 Hz
# detrended floor), i.e. one number explains both the residual's offset
# AND its slope simultaneously. For every other station tried (Hh, Cd, Ke),
# no single offset does this. Fixing that would require also fitting a drift (rate) and possibly a quadratic
# (acceleration) term to fully explain the trend, i.e. more than a pure
# constant clock/timestamp shift can produce.
TIME_SHIFT_BY_STATION_S = {
    "Yg": 0.0, # optimal: 0.84 s removes most of linear residual trend
    "Hb": 0.0, 
    "Hh": 0.0, 
    "Cd": 0.0, 
    "Ke": 0.0,
}

# JUICE's Lunar-Earth gravity assist spans two separate flybys on consecutive
# days, each tracked by Hh in its own FDETS file: the lunar flyby on 2024-08-19
# (closest approach to the Moon at 21:16 UTC)
# and the separate Earth flyby the following day, 2024-08-20. The Moon's radio
# occultation of JUICE during the lunar flyby forces NWNORCIA's uplink from
# "two-way"(= three way in out convention) into one-way mode; per Molera Calves et al. (2026) Fig. 10, Hh's
# "two-way"(= three way in out convention) tracking switches to one-way at 20:30 UTC and resumes two-way at
# 21:13 UTC, bracketing signal ingress (20:38) and egress (21:08). Only the
# "two-way"(= three way in out convention) segment is usable with this script's three-way link model. 
# This segmentation is Hh-specific (see build_passes_for_station).
lunar_occultation_start = DateTime.from_python_datetime(datetime(2024, 8, 19, 20, 30)).to_epoch()
lunar_occultation_end = DateTime.from_python_datetime(datetime(2024, 8, 19, 21, 14)).to_epoch()

# Frequency-reference jump specific to the Hh lunar-flyby pass: confirmed
# against the raw FDETS file to coincide exactly with a scan-index boundary
# (scan 0001 ends 19:30:56, scan 0002 begins 19:31:06). This looks like an inter-scan
# calibration/reference discontinuity in the FDETS/correlator processing.
# The lunar-flyby pass is split into two separate
# passes here (pre-/post-scan jump), so each is a single, internally-consistent
# segment. 
hh_jump_time = DateTime.from_python_datetime(datetime(2024, 8, 19, 19, 31)).to_epoch()

# Environment time span, wide enough to cover every station's observations
# (earliest: Cd's 2024-08-19 file, from ~15:48 UTC; latest: every station's
# 2024-08-20 file ends by ~19:28 UTC).
environment_start_time = DateTime.from_python_datetime(datetime(2024, 8, 19, 10, 58, 36)).to_epoch()
environment_end_time = DateTime.from_python_datetime(datetime(2024, 8, 20, 19, 28, 9)).to_epoch()
environment_start_time_buffer = environment_start_time - 86400
environment_end_time_buffer = environment_end_time + 86400

def build_passes_for_station(station_id):
    """Every station gets the same top-level Pre_Lunar_Occultation /
    Pre_Earth_Occultation split, each mapped to that station's own day-19 /
    day-20 FDETS file. Only Hh further splits Pre_Lunar_Occultation into its
    scan-jump halves (see the module-level comment on hh_jump_time for why);
    every other station's lunar-flyby data is kept as one unsegmented pass,
    since the scan-jump bound was only established for Hh.
    """
    if station_id == "Hh":
        return [
            {
                "label": "Pre_Lunar_Occultation_Pre_Scan_Jump",
                "fdets_glob": f"Fdets.jui2024.08.19.{station_id}.r*i.txt",
                "keep_start_time": environment_start_time,
                "keep_end_time": hh_jump_time,
            },
            {
                "label": "Pre_Lunar_Occultation_Post_Scan_Jump",
                "fdets_glob": f"Fdets.jui2024.08.19.{station_id}.r*i.txt",
                "keep_start_time": hh_jump_time,
                "keep_end_time": lunar_occultation_start,
            },
            {
                "label": "Pre_Earth_Occultation",
                "fdets_glob": f"Fdets.jui2024.08.20.{station_id}.r*i.txt",
                "keep_start_time": lunar_occultation_end,
                "keep_end_time": environment_end_time,
            },
        ]
    return [
        {
            "label": "Pre_Lunar_Occultation",
            "fdets_glob": f"Fdets.jui2024.08.19.{station_id}.r*i.txt",
            "keep_start_time": environment_start_time,
            "keep_end_time": lunar_occultation_start,
        },
        {
            "label": "Pre_Earth_Occultation",
            "fdets_glob": f"Fdets.jui2024.08.20.{station_id}.r*i.txt",
            "keep_start_time": lunar_occultation_end,
            "keep_end_time": environment_end_time,
        },
    ]

# %% [markdown]
# ## 2. Helper Functions
#
# A quick reference for the non-obvious pieces used below:
#
# - **FDETS file format.** Each data line is either 5 or 6 whitespace-
#   separated columns: an optional leading *scan number*, a UTC datetime
#   string, signal-to-noise ratio, normalised spectral maximum, the
#   measured Doppler frequency detection (Hz, relative to a header-declared
#   *base frequency*), and the Doppler noise. A "scan" is one correlator
#   processing chunk; scan boundaries are worth marking on plots because
#   inter-scan calibration jumps could be a real source of
#   artifacts (see Hh's scan-jump split above).
# - **`FdetDateFormat`.** tudatpy's `observations_from_fdets_files` used to
#   take an explicit list of column-type strings; that form is deprecated
#   in favor of `date_format=FdetDateFormat.datetime_string`, which also
#   auto-detects an optional leading scan-number column instead of
#   requiring it to be stripped out beforehand.
# - **Linear detrending.** `scipy.signal.detrend` fits and subtracts the
#   best-fit straight line from the residual. If what's left over is small
#   and unstructured, the *original* residual's Hz-level discrepancy was
#   well described by just two numbers, i.e. a constant offset and a constant
#   drift rate. 
# - **Doppler → line-of-sight velocity/acceleration.** A Doppler frequency
#   offset $\Delta f$ maps to an equivalent line-of-sight velocity error via
#   the one-way relation $\Delta v = c\,\Delta f / f_0$; applying the same
#   relation to the residual's *slope* (Hz/s) gives an equivalent
#   line-of-sight *acceleration* error. This allows us to turn the offset and slope from the linear detrending into a physically interpretable level of unexplained velocity and acceleration. This is useful to get an idea on the order of magnitude difference between the real orbit and the kernel's orbital solution. However, running this for multiple stations, it looks like we cannot reliably give a single order of magnitude, since different stations yield different velocity discrepancies (differing by one or two order of magnitude, both on the pre-Lunar and pre-Earth occultation passes).

# %%
SPEED_OF_LIGHT = 299792458.0  # m/s

# Plot styling shared by every figure produced in %% 5-8.
COLOR_OBS = '#4C72B0'
COLOR_SIM = '#DD8452'
COLOR_RES = '#55A868'
COLOR_RAW = '#999999'
COLOR_SCAN_BOUNDARY = '#888888'

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

def remap_vmf_station_names(input_files, output_folder, station_name_mapping):
    """Keeps only the VMF entries for the given stations, renaming their VMF
    site codes (e.g. 'HRAO') to the ground station names used in `bodies`
    (e.g. 'Hh'), since set_vmf_troposphere_data() only matches by exact
    string equality against the ground station map -- it has no notion that
    a VMF/IGS site code and our internal station name refer to the same
    physical antenna."""
    os.makedirs(output_folder, exist_ok=True)
    output_paths = []
    for input_file in input_files:
        output_path = os.path.join(output_folder, os.path.basename(input_file))
        with open(input_file, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                if line.startswith('#') or not line.strip():
                    f_out.write(line)
                    continue
                parts = line.split()
                if parts and parts[0] in station_name_mapping:
                    parts[0] = station_name_mapping[parts[0]]
                    f_out.write(" ".join(parts) + '\n')
        output_paths.append(output_path)
    return output_paths

def fit_linear_residual_trend(times_tdb, residuals):
    """Best-fit line residual(t) ~= intercept + slope*(t - times_tdb[0]), on
    the raw (undetrended) residual. Returns (intercept_hz, slope_hz_per_s)."""
    t_relative = times_tdb - times_tdb[0]
    slope, intercept = np.polyfit(t_relative, residuals, 1)
    return intercept, slope

def estimate_los_velocity_and_acceleration_error(intercept_hz, slope_hz_per_s, carrier_freq_hz):
    """Converts a Doppler residual's linear trend into an equivalent
    line-of-sight velocity/acceleration error via the one-way Doppler relation
    Delta_v = c * Delta_f / f0 (and, since acceleration is the time-derivative
    of velocity, the same relation applied to the residual's slope gives
    Delta_a = c * slope / f0). This is a simple, one-way-equivalent estimate:
    for a true three-way link (uplink and downlink legs geometrically
    distinct, as here) the real sensitivity can go up to ~2x if both
    legs contribute comparably, so treat these as conservative/order-of-
    magnitude numbers, not a precise velocity/acceleration measurement.
    """
    delta_v = SPEED_OF_LIGHT * intercept_hz / carrier_freq_hz
    delta_a = SPEED_OF_LIGHT * slope_hz_per_s / carrier_freq_hz
    return delta_v, delta_a

def shift_fdets_timestamps(input_file, output_folder, time_shift_seconds=0.0):
    """Shifts the UTC time tag by the given number of seconds, leaving every
    other column untouched, including an optional leading scan-number
    column, if present. The column count is detected per line (5 fields:
    no scan number; 6 fields: leading scan number).
    """
    filename = os.path.basename(input_file)
    output_path = os.path.join(output_folder, filename)
    os.makedirs(output_folder, exist_ok=True)
    with open(input_file, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('#') or not line.strip():
                f_out.write(line)
            else:
                parts = line.split()
                if len(parts) not in (5, 6):
                    continue
                utc_index = 1 if len(parts) == 6 else 0
                utc_str = parts[utc_index]
                fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in utc_str else "%Y-%m-%dT%H:%M:%S"
                dt = datetime.strptime(utc_str, fmt)
                dt_shifted = dt + timedelta(seconds=time_shift_seconds)
                parts[utc_index] = dt_shifted.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                f_out.write(" ".join(parts) + '\n')
    return output_path

def get_scan_boundary_datetimes(fdets_file):
    """Parses the (already optionally timestamp-shifted) FDETS file's scan-number and
    UTC-time columns, and returns a list of UTC datetimes marking every
    internal scan-to-scan transition: the timestamp of the last observation
    of the ending scan, immediately followed by the timestamp of the first
    observation of the next scan. The very first scan's start and the very
    last scan's end aren't included -- those coincide with the plot's own
    time axis limits. Returns an empty list if the file has no scan-number
    column.
    """
    scan_numbers = []
    utc_datetimes = []
    with open(fdets_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 6:
                scan_numbers.append(parts[0])
                utc_str = parts[1]
            elif len(parts) == 5:
                scan_numbers.append(None)
                utc_str = parts[0]
            else:
                continue
            fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in utc_str else "%Y-%m-%dT%H:%M:%S"
            utc_datetimes.append(datetime.strptime(utc_str, fmt))

    if not scan_numbers or scan_numbers[0] is None:
        return []

    boundary_datetimes = []
    for i in range(1, len(scan_numbers)):
        if scan_numbers[i] != scan_numbers[i - 1]:
            boundary_datetimes.append(utc_datetimes[i - 1])
            boundary_datetimes.append(utc_datetimes[i])
    return boundary_datetimes

def load_pass_residuals(pass_config, simulators, bodies, station_positions, station_id, time_shift):
    """Reads this pass's FDETS file, computes residuals against the given
    simulators, and returns (times_tdb, residuals, observations_val,
    simulated_val, current_base_freq, scan_boundary_datetimes), all filtered
    to this pass's time window. Re-reading the FDETS file each call (rather
    than reusing a collection) keeps this safe to call again for a different
    simulator set, since compute_residuals_and_dependent_variables mutates
    its observation collection in place.
    """
    fdets_files = glob.glob(os.path.join(fdets_folder, pass_config["fdets_glob"]))
    if not fdets_files:
        return None
    f_path = fdets_files[0]
    current_base_freq = extract_base_frequency(f_path)
    fdets_clean = shift_fdets_timestamps(f_path, output_folder, time_shift)
    scan_boundary_datetimes = get_scan_boundary_datetimes(fdets_clean)

    fdets_collection = observations.observations_from_fdets_files(
        fdets_clean, current_base_freq, FdetDateFormat.datetime_string,
        "JUICE", 'NWNORCIA', station_id,
        observations_setup.ancillary_settings.FrequencyBands.x_band,
        observations_setup.ancillary_settings.FrequencyBands.x_band, station_positions
    )
    estimation.observations.compute_residuals_and_dependent_variables(fdets_collection, simulators, bodies)

    pass_filter = observations.observations_processing.observation_filter(
        observations.observations_processing.ObservationFilterType.time_bounds_filtering,
        pass_config["keep_start_time"],
        pass_config["keep_end_time"],
        use_opposite_condition=True,
    )
    fdets_collection.filter_observations(pass_filter)

    times_tdb = np.asarray(fdets_collection.get_concatenated_observation_times())
    if times_tdb.size == 0:
        return None
    residuals = np.asarray(fdets_collection.get_concatenated_residuals())
    observations_val = np.asarray(fdets_collection.get_concatenated_observations())
    simulated_val = observations_val - residuals
    return times_tdb, residuals, observations_val, simulated_val, current_base_freq, scan_boundary_datetimes

def times_tdb_to_datetime(times_tdb, converter):
    """Converts an array of TDB epochs to Python datetimes (UTC) for plotting."""
    return np.array([
        DateTime.to_python_datetime(
            DateTime.from_epoch(
                converter.convert_time(time_representation.tdb_scale, time_representation.utc_scale, t)
            )
        )
        for t in times_tdb
    ])

def _setup_figure():
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={'height_ratios': [1, 1]}
    )
    plt.subplots_adjust(hspace=0.08)
    return fig, ax_top, ax_bottom

def _plot_observed_vs_simulated(ax_top, times_datetime, result):
    ax_top.plot(times_datetime, result["observations_val"],
                label='Observed', color=COLOR_OBS, linewidth=2)
    ax_top.plot(times_datetime, result["simulated_val"],
                label='Simulated', color=COLOR_SIM, linestyle='--', linewidth=2)
    ax_top.set_ylabel('Frequency [Hz]')
    ax_top.legend(loc='lower right')

def _draw_scan_boundaries(ax_top, ax_bottom, times_datetime, scan_boundary_datetimes):
    """Draws thin dashed vertical lines at every scan start/end within the
    plotted time range."""
    if len(times_datetime) == 0:
        return
    t_min, t_max = times_datetime.min(), times_datetime.max()
    visible_boundaries = [t for t in scan_boundary_datetimes if t_min <= t <= t_max]
    for i, t in enumerate(visible_boundaries):
        ax_top.axvline(t, color=COLOR_SCAN_BOUNDARY, linestyle=':', linewidth=0.8, alpha=0.5)
        ax_bottom.axvline(t, color=COLOR_SCAN_BOUNDARY, linestyle=':', linewidth=0.8, alpha=0.5,
                          label='Scan boundary' if i == 0 else None)

def _finish_axes(ax_top, ax_bottom, rms_text):
    ax_bottom.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax_bottom.set_ylabel('Residual [Hz]')
    ax_bottom.set_xlabel('Time (UTC)')
    ax_bottom.text(
        0.01, 0.95, rms_text,
        transform=ax_bottom.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    ax_bottom.legend(loc='upper right')
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_bottom.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax_bottom.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
    for ax in [ax_top, ax_bottom]:
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

def _save_and_close(fig, pass_config, result, filename_suffix, title_suffix, station_id, site_full_name, subfolder):
    fig.suptitle(
        f"JUICE Doppler Residuals ({pass_config['label'].replace('_', ' ')}){title_suffix}\n"
        f"{station_id} ({site_full_name}) — {result['current_base_freq']/1e6:.2f} MHz",
        fontsize=15,
        y=0.98
    )
    output_dir = os.path.join(plots_folder, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{pass_config['label']}_{station_id}{filename_suffix}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

# %% [markdown]
# ## 3. Environment Setup
#
# This builds the tudatpy `SystemOfBodies` used for every station's
# simulation: JUICE's trajectory (from an SPK kernel), the planets (for
# light-time/relativistic corrections), Earth's rotation model, every
# station's geodetic position, and the light-time correction models
# (relativistic + optionally tropospheric).
#
# **The three-way link.** For every station, `LinkDefinition` wires up
# receiver (that station) → reflector (JUICE, which transponds the uplinked
# signal) → transmitter (NWNORCIA). The Doppler *measured frequency*
# observable model then needs to know NWNORCIA's actual uplinked frequency
# at each epoch, and that's what `nominal_station_ramp` provides.

# %%
site_full_name_by_station = {}
for station_id in STATION_IDS:
    site_full_name = constants.ID_TO_SITE.get(station_id)
    if not site_full_name or site_full_name not in constants.STATION_GEODETIC_POSITIONS:
        raise ValueError(f"No geodetic position known for station '{station_id}'.")
    site_full_name_by_station[station_id] = site_full_name

spice.load_standard_kernels()
spice.load_kernel(os.path.join(kernels_folder, "spk/juice_orbc_000097_230414_310721_v02.bsp"))
spice.load_kernel(os.path.join(kernels_folder, "spk/inpop19a_19900101_20500101.bsp"))

body_settings = environment_setup.get_default_body_settings_time_limited(
    ["Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon"],
    environment_start_time, environment_end_time, "Earth", "J2000")

# Global frame origin must stay "Earth", not "SSB". Using "SSB" reproduced as VMF3 blowing up at some epochs.
global_frame_origin = "Earth"; global_frame_orientation = "J2000"
body_settings.get('Earth').shape_settings = environment_setup.shape.oblate_spherical_spice()
body_settings.get('Earth').gravity_field_settings.associated_reference_frame = "ITRS"

spacecraft_name = "JUICE"; spacecraft_central_body = "Earth"
body_settings.add_empty_settings(spacecraft_name)
body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
    environment_start_time_buffer, environment_end_time_buffer, 1.0, spacecraft_central_body, global_frame_orientation)
body_settings.get(spacecraft_name).rotation_model_settings = environment_setup.rotation_model.spice(
    global_frame_orientation, spacecraft_name + "_SPACECRAFT", "")

ground_station_settings = environment_setup.ground_station.radio_telescope_stations()
for station_id in STATION_IDS:
    coords = constants.STATION_GEODETIC_POSITIONS[site_full_name_by_station[station_id]]
    tudat_coords = [coords[0], np.deg2rad(coords[1]), np.deg2rad(coords[2])]
    ground_station_settings = ground_station_settings + [
        environment_setup.ground_station.basic_station(station_id, tudat_coords, element_conversion.geodetic_position_type)
    ]
body_settings.get('Earth').ground_station_settings = ground_station_settings

body_settings.get('Earth').rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
    environment_setup.rotation_model.iau_2006, "J2000",
    interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), environment_start_time_buffer, environment_end_time_buffer, 3600.0),
    interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), environment_start_time_buffer, environment_end_time_buffer, 3600.0),
    interpolators.interpolator_generation_settings(interpolators.cubic_spline_interpolation(), environment_start_time_buffer, environment_end_time_buffer, 10.0))

bodies = environment_setup.create_system_of_bodies(body_settings)

vehicleSys = environment.VehicleSystems()
vehicleSys.set_default_transponder_turnaround_ratio_function()
bodies.get_body("JUICE").system_models = vehicleSys

converter = time_representation.default_time_scale_converter()

# Norcia Ramp: segment 1 covers the lunar-flyby pass (2024-08-19), segment 2
# covers the Earth-flyby pass (2024-08-20). Shared by every station, since
# NWNORCIA is the single uplink transmitter for all of them.
ramp_seg1_start = DateTime(2024, 8, 19, 10, 45, 36).epoch()
ramp_seg1_end = DateTime(2024, 8, 20, 11, 32, 17).epoch()
ramp_seg2_start = ramp_seg1_end
ramp_seg2_end = DateTime(2024, 8, 22, 20, 55, 9).epoch()
nominal_seg1_freq = 7180.142419e6
nominal_seg2_freq = 7180.127320e6

nominal_station_ramp = environment.PiecewiseLinearFrequencyInterpolator(
    [ramp_seg1_start, ramp_seg2_start], [ramp_seg1_end, ramp_seg2_end],
    [0, 0], [nominal_seg1_freq, nominal_seg2_freq]
)
bodies.get('Earth').get_ground_station('NWNORCIA').transmitting_frequency_calculator = nominal_station_ramp

base_light_time_corrections = [
    observable_models_setup.light_time_corrections.first_order_relativistic_light_time_correction(["Sun", "Earth", "Moon"])
]

# VMF3 tropospheric correction data is attached to the stations ONCE, so it's
# available to whichever simulator set (below) actually includes the
# correction in its light_time_correction_list.
#
# The VMF files use IGS/geodetic-network site codes (e.g. 'HRAO', 'CEDU',
# 'NNOR'), not our internal station names ('Hh', 'Cd', 'NWNORCIA'). 
# set_vmf_troposphere_data() only matches ground stations by exact name, so
# without this remapping it silently attaches data to none of the stations
# actually used in this script's links.
vmf_station_name_mapping = {
    "HRAO": "Hh",
    "CEDU": "Cd",
    "NNOR": "NWNORCIA",
    "YARR": "Yg",
    "HOB2": "Hb",
    "KAT1": "Ke",
}
vmf_files_remapped = remap_vmf_station_names(
    ["juice_archive/vmf/2024231.vmf3_g", "juice_archive/vmf/2024232.vmf3_g",
     "juice_archive/vmf/2024233.vmf3_g", "juice_archive/vmf/2024234.vmf3_g"],
    "juice_archive/vmf_remapped", vmf_station_name_mapping)
observable_models_setup.light_time_corrections.set_vmf_troposphere_data(
    vmf_files_remapped, True, False, bodies, True, True)

light_time_correction_list_no_vmf = list(base_light_time_corrections)
light_time_correction_list_with_vmf = base_light_time_corrections + [
    observable_models_setup.light_time_corrections.vmf3_tropospheric_light_time_correction(use_gradient_correction=False)
]

# Per-station link definition, station positions, and (no-VMF, with-VMF)
# simulator pair. 
norcia_position = bodies.get('Earth').get_ground_station('NWNORCIA').station_state.get_cartesian_position(environment_start_time)

link_definition_by_station = {}
station_positions_by_station = {}
simulators_no_vmf_by_station = {}
simulators_with_vmf_by_station = {}

for station_id in STATION_IDS:
    link_definition = observable_models_setup.links.LinkDefinition({
        observable_models_setup.links.receiver: observable_models_setup.links.body_reference_point_link_end_id('Earth', station_id),
        observable_models_setup.links.reflector1: observable_models_setup.links.body_origin_link_end_id('JUICE'),
        observable_models_setup.links.transmitter: observable_models_setup.links.body_reference_point_link_end_id('Earth', 'NWNORCIA'),
    })
    link_definition_by_station[station_id] = link_definition

    station_positions_by_station[station_id] = {
        station_id: bodies.get('Earth').get_ground_station(station_id).station_state.get_cartesian_position(environment_start_time),
        'NWNORCIA': norcia_position,
    }

    observable_settings_no_vmf = [observable_models_setup.model_settings.doppler_measured_frequency(link_definition, light_time_correction_list_no_vmf)]
    observable_settings_with_vmf = [observable_models_setup.model_settings.doppler_measured_frequency(link_definition, light_time_correction_list_with_vmf)]
    simulators_no_vmf_by_station[station_id] = observations_setup.observations_simulation_settings.create_observation_simulators(observable_settings_no_vmf, bodies)
    simulators_with_vmf_by_station[station_id] = observations_setup.observations_simulation_settings.create_observation_simulators(observable_settings_with_vmf, bodies)

# %% [markdown]
# ## 4. Compute Residuals
#
# For every station and every one of its passes, this loop reads the FDETS
# file(s), builds the corresponding tudatpy observation collection, and
# computes the O-C residual against both the no-VMF and with-VMF
# simulators. The results are stashed in nested dictionaries
# (`results_no_vmf[station_id][pass_label]`) so the four plotting sections
# below can just look them up rather than recomputing anything.
#
# If you don't see a plot for a given station/pass combination later on,
# check the printed messages from this cell first — a missing FDETS file
# (e.g. Yg/Hb/Ke have no 2024-08-19 file) is reported here, not silently
# skipped.

# %%
passes_by_station = {station_id: build_passes_for_station(station_id) for station_id in STATION_IDS}
results_no_vmf = {station_id: {} for station_id in STATION_IDS}
results_with_vmf = {station_id: {} for station_id in STATION_IDS}

for station_id in STATION_IDS:
    for pass_config in passes_by_station[station_id]:
        time_shift = TIME_SHIFT_BY_STATION_S[station_id]
        result_no_vmf = load_pass_residuals(
            pass_config, simulators_no_vmf_by_station[station_id], bodies,
            station_positions_by_station[station_id], station_id, time_shift)
        result_with_vmf = load_pass_residuals(
            pass_config, simulators_with_vmf_by_station[station_id], bodies,
            station_positions_by_station[station_id], station_id, time_shift)
        if result_no_vmf is None or result_with_vmf is None:
            print(f"No FDETS file found for {station_id} pass '{pass_config['label']}' ({pass_config['fdets_glob']}), skipping.")
            continue

        for results_dict, result in ((results_no_vmf[station_id], result_no_vmf), (results_with_vmf[station_id], result_with_vmf)):
            times_tdb, residuals, observations_val, simulated_val, current_base_freq, scan_boundaries = result
            results_dict[pass_config["label"]] = {
                "pass_config": pass_config,
                "times_tdb": times_tdb,
                "residuals": residuals,
                "observations_val": observations_val,
                "simulated_val": simulated_val,
                "current_base_freq": current_base_freq,
                "scan_boundaries": scan_boundaries,
            }

# %% [markdown]
# ## 5. Plots: Original residuals, no tropospheric correction
#
# The rawest view of the data: observed vs. simulated frequency on top,
# the O-C residual (with scan boundaries marked) on the bottom. This is the
# right first plot to look at for any new station/pass, before applying
# tropospheric corrections or detrending, does the residual look flat and noisy
# (good sign), or does it have visible structure (offset, slope, curvature,
# or a jump at a scan boundary)?
#
# Output goes to `plots_folder/no_vmf/`.

# %%
for station_id in STATION_IDS:
    for pass_config in passes_by_station[station_id]:
        if pass_config["label"] not in results_no_vmf[station_id]:
            continue
        result = results_no_vmf[station_id][pass_config["label"]]
        times_datetime = times_tdb_to_datetime(result["times_tdb"], converter)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax_top, ax_bottom = _setup_figure()
        _plot_observed_vs_simulated(ax_top, times_datetime, result)
        _draw_scan_boundaries(ax_top, ax_bottom, times_datetime, result["scan_boundaries"])

        ax_bottom.scatter(times_datetime, result["residuals"],
                           color=COLOR_RAW, s=7, alpha=0.8, label='Original')

        rms_text = f"RMS (Original): {np.sqrt(np.mean(result['residuals']**2)):.4f} Hz"
        _finish_axes(ax_top, ax_bottom, rms_text)
        _save_and_close(fig, pass_config, result, "_Original_NoVMF", " — Original, no tropospheric correction",
                        station_id, site_full_name_by_station[station_id], "no_vmf")

# %% [markdown]
# ## 6. Plots: Original residuals, with VMF3 tropospheric correction
#
# The same plot as above, but with the VMF3 tropospheric delay correction
# included in the light-time computation. Compare the RMS annotation
# against the no-VMF version for the same station/pass: a large improvement
# means the uncorrected residual had genuine tropospheric-shaped structure
# in it; a negligible or even slightly *worse* RMS means the troposphere
# isn't the dominant error source for that pass, hence some other, larger effect
# is masking whatever small, real contribution VMF3 makes. 
#
# Output goes to `plots_folder/vmf/`.

# %%
for station_id in STATION_IDS:
    for pass_config in passes_by_station[station_id]:
        if pass_config["label"] not in results_with_vmf[station_id]:
            continue
        result = results_with_vmf[station_id][pass_config["label"]]
        times_datetime = times_tdb_to_datetime(result["times_tdb"], converter)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax_top, ax_bottom = _setup_figure()
        _plot_observed_vs_simulated(ax_top, times_datetime, result)
        _draw_scan_boundaries(ax_top, ax_bottom, times_datetime, result["scan_boundaries"])

        ax_bottom.scatter(times_datetime, result["residuals"],
                           color=COLOR_RAW, s=7, alpha=0.8, label='Original')

        rms_text = f"RMS (Original): {np.sqrt(np.mean(result['residuals']**2)):.4f} Hz"
        _finish_axes(ax_top, ax_bottom, rms_text)
        _save_and_close(fig, pass_config, result, "_Original_VMF", " — Original, VMF3 tropospheric correction",
                        station_id, site_full_name_by_station[station_id], "vmf")

# %% [markdown]
# ## 7-8. Plots: Original vs Detrended, with LOS velocity/acceleration estimate
#
# This is where the diagnostic payoff is. Each plot shows the raw residual,
# its best-fit straight line, and the detrended residual together, plus a
# text box reporting:
#
# - the fitted intercept and slope of that straight line,
# - the equivalent line-of-sight velocity and acceleration error those
#   imply (see the Section 2 note on the Doppler-to-velocity relation),
# - RMS before and after detrending.
#
# **How to read this across stations.** If the detrended RMS drops to a
# clean, small noise floor, the original Hz-level discrepancy was well
# explained by just an offset and a drift rate, and it is worth asking *what*
# produces such a clean two-parameter trend (a timing bias is one
# candidate; see the `TIME_SHIFT_BY_STATION_S` discussion in Section 1 for
# how to test that hypothesis for a given station (e.g. Yarragadee)). If the detrended
# residual still has visible curvature or a large floor, the story is more
# complex than "the trajectory/timing is slightly off by a constant amount". 
# In that case, go back to Sections 5/6 and look for data processing and calibration artifacts,
# or check whether VMF3 helps.
#
# Two variants are produced: `no_vmf_detrended/` (detrending applied to the
# no-VMF residual) and `vmf_detrended/` (same, with VMF3 applied first).

# %%
def plot_original_vs_detrended(results_by_station, filename_suffix, title_suffix, subfolder):
    for station_id in STATION_IDS:
        results_by_label = results_by_station[station_id]
        for pass_config in passes_by_station[station_id]:
            if pass_config["label"] not in results_by_label:
                continue
            result = results_by_label[pass_config["label"]]
            times_tdb = result["times_tdb"]
            residuals = result["residuals"]
            residuals_detrended = signal.detrend(residuals, bp=0)

            intercept_hz, slope_hz_per_s = fit_linear_residual_trend(times_tdb, residuals)
            linear_fit_curve = intercept_hz + slope_hz_per_s * (times_tdb - times_tdb[0])
            delta_v_los, delta_a_los = estimate_los_velocity_and_acceleration_error(
                intercept_hz, slope_hz_per_s, result["current_base_freq"])

            times_datetime = times_tdb_to_datetime(times_tdb, converter)

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax_top, ax_bottom = _setup_figure()
            _plot_observed_vs_simulated(ax_top, times_datetime, result)
            _draw_scan_boundaries(ax_top, ax_bottom, times_datetime, result["scan_boundaries"])

            ax_bottom.scatter(times_datetime, residuals,
                               color=COLOR_RAW, s=7, alpha=0.8, label='Original')
            ax_bottom.plot(times_datetime, linear_fit_curve,
                            color='black', linewidth=1.5, label='Linear fit')
            ax_bottom.scatter(times_datetime, residuals_detrended,
                               s=7, color=COLOR_RES, label='Detrended')

            rms_text = (
                f"RMS (Original): {np.sqrt(np.mean(residuals**2)):.4f} Hz\n"
                f"RMS (Detrended): {np.sqrt(np.mean(residuals_detrended**2)):.4f} Hz\n"
                f"Linear fit: intercept={intercept_hz:+.4f} Hz, slope={slope_hz_per_s:+.2e} Hz/s\n"
                f"LOS vel. error~{delta_v_los:+.4f} m/s, LOS accel. error~{delta_a_los:+.2e} m/s^2"
            )
            _finish_axes(ax_top, ax_bottom, rms_text)
            _save_and_close(fig, pass_config, result, filename_suffix, title_suffix,
                            station_id, site_full_name_by_station[station_id], subfolder)

plot_original_vs_detrended(results_no_vmf, "_Detrended", " — Original vs Detrended", "no_vmf_detrended")
plot_original_vs_detrended(results_with_vmf, "_Detrended_VMF", " — Original vs Detrended, VMF3 tropospheric correction", "vmf_detrended")

# %% [markdown]
# ## Exercises
#
# A few things worth trying, using the tools already built above:
#
# 1. **Timing-bias hypothesis test.** Pick a station/pass, sweep
#    `TIME_SHIFT_BY_STATION_S[<station>]` over a range of values (e.g.
#    `numpy.arange(-5, 5, 0.1)`), and plot RMS(Original) and RMS(Detrended)
#    against the shift. Does a single value collapse *both* down to the
#    same clean floor (a timing-artifact signature, like Yg), or does no
#    single value do that (pointing to something other than pure timing)?
# 2. **Scan-boundary artifacts elsewhere.** Every plot marks scan
#    boundaries. Do any of the non-Hh stations show a jump at one, the way
#    Hh does on 2024-08-19? If so, that station might benefit from the same
#    kind of pass-splitting Hh gets.
