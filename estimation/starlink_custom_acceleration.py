# Load required standard modules
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import linalg

import matplotlib
matplotlib.use('macosx')
from matplotlib import pyplot as plt


# Load required tudatpy modules
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import frame_conversion
from tudatpy.util import result2array


"""
This demo shall demonstrate the flexibility of tudatpy and its ability to closely mimic operational scenarios.
In this demo, the user is given the role of the operator of a commercial LEO satellite. 
It is assumed that the operator is given the task to perform a multi-segment low-thrust collision avoidance manoeuvre.
Without thrust vector control (would be highly unusual for low-thrust station keeping), the thrusting direction during the manoeuvre is determined by the s/c attitude. 
We assume that the operator is given a manoeuvre design and corresponding rotational ephemeris of the sc and is now conducting a simulation of the development of sc state uncertainty over the course of the manoeuvre.
"""


########################################################################################################################
############## CUSTOM CLASS DEFINITIONS   ##############################################################################
# For the convenient implementation of the functionality described above, we define 3 custom classes:

# 1) Custom spacecraft orientation ephemeris
"""
The direction of thrust follows from the rotational ephemeris that is defined a-priori for the spacecraft.
Via the MySpacecraftOrientation class, the rotational state as a function of time is defined. 
In this example, the user defines the rotational ephemeris relative to the tnw orboit defined attitude of the s/c.
In principle, the rotation can be defined w.r.t. to inertial or other state-derived references too.
Any state-dependent definition of the rotational ephemeris (i.e. where tudat envionment info is queried when rotational state is called) 
must be facilitated by a class, for which the self.bodies member (holding the environment information at runtime) can be assigned dynamically.
"""

class MySpacecraftOrientation:

    def __init__(self, sc_name):

        self.sc_name = sc_name
        self.bodies = None

        self.inertial_to_body_fixed_callable = None
        self.body_fixed_to_inertial_callable = None


    def set_bodies(self, bodies):
        self.bodies = bodies

    def set_sc_rotation_ephemeris_in_tnw(self, rotation_matrix_callable):

        def inertial_to_body_fixed_callable(epoch):

            translational_state = self.bodies.get_body(self.sc_name).state
            R1 = frame_conversion.inertial_to_tnw_rotation_matrix(translational_state)
            R2 = rotation_matrix_callable(epoch)

            return np.matmul(R2, R1)

        self.inertial_to_body_fixed_callable = inertial_to_body_fixed_callable

        def body_fixed_to_inertial_callable(epoch):

            return self.inertial_to_body_fixed_callable(epoch).T

        self.body_fixed_to_inertial_callable = body_fixed_to_inertial_callable



# 2) Custom Manoeuvre class
"""
This class is a collection for thrusting segments objects (see below), which is used to consolidate the user-defined segments 
of any multi-segment low-thrust manoeuvre and which facilitates "cumulative" operations over all segments.
This is necessary, because the custom acceleration interface of the propagator requires *one* acceleration callable and it would therefore not be possible to pass the acceleration callable of each thrusting segments to the propagator settings.
Force model calculations (as well as definition of custom estimatable parameters) however are done at the "Segment" level (see below).
"""

class MyManoeuvre:
    def __init__(self):

        self.segments = []

    def add_segments(self, new_segments):

        ###
        if len(self.segments) > 0:
            old_segments = self.segments
            segments = old_segments + new_segments
            ordered_segments = sorted(segments, key=lambda x: x.start_time)
            self.segments = ordered_segments

        else:
            self.segments = new_segments


    def set_bodies(self, bodies):

        for segment in self.segments:
            segment.set_bodies(bodies)


    def consolidated_custom_acceleration_callable(self, epoch):

        acceleration = np.zeros(3)
        for segment in self.segments:
            acceleration += segment.compute_acceleration_in_inertial(epoch)

        return acceleration

    def get_thrust_magnitude_parameter_variance_vector(self):

        parameter_variance_vector = np.zeros(len(self.segments))
        for i, segment in enumerate(self.segments):
            parameter_variance_vector[i] = segment.thrust_magnitude_variance

        return parameter_variance_vector



# 2) Custom Thrusting Segment class

"""
The thrusting intervals of the custom manoeuvre are defined as so-called segments via the ConstantThrustMagnitudeSegment class. 
For each manoeuvre segment one instance of this class is created, which then handles the custom acceleration model computations for the given segment.
This is also where the partials for estimatable parameters of the custom acceleration model must be defined.
Segment objects are then consolidated in a single MyManoeuvre object.
"""

class ConstantThrustMagnitudeSegment:
    def __init__(self, start_time, end_time, sc_name):

        self.sc_name = sc_name
        self.bodies = None

        self.start_time = start_time
        self.end_time = end_time

        self.thrust_magnitude = None
        self.thrust_unit_vector_body_fixed = None
        self.thrust_magnitude_variance = None


    ####### setters #############

    def set_bodies(self, bodies):
        self.bodies = bodies

    def set_thruster_properties(self, magnitude, vector_body_fixed, magnitude_variance):
        self.thrust_magnitude = magnitude
        self.thrust_unit_vector_body_fixed = vector_body_fixed
        self.thrust_magnitude_variance = magnitude_variance


    ####### model computations #########
    def get_body_fixed_thrust_vector(self, epoch):

        if self.start_time <= epoch <= self.end_time:
            return self.thrust_unit_vector_body_fixed * self.thrust_magnitude
        else:
            return np.zeros(3)

    def get_sc_mass_from_environment(self):
        return self.bodies.get(self.sc_name).mass

    def compute_force_in_inertial(self, epoch):

        body_fixed_thrust_vector = self.get_body_fixed_thrust_vector(epoch)
        R = self.bodies.get(self.sc_name).body_fixed_to_inertial_frame

        return R.dot(body_fixed_thrust_vector)

    def compute_acceleration_in_inertial(self, epoch):
        return self.compute_force_in_inertial(epoch) / self.get_sc_mass_from_environment()


    ####### parameters  #########

    # setter function for thrust magnitude parameter
    def set_thrust_magnitude(self, thrust_magnitude):
        self.thrust_magnitude = thrust_magnitude

    # getter function for thrust magnitude parameter
    def get_thrust_magnitude(self):
        return np.array([self.thrust_magnitude])


    ####### partials  #########
    def force_model_partial_to_thrust_magnitude(self, epoch, dummy):

        if self.start_time <= epoch <= self.end_time:
            partial = self.compute_acceleration_in_inertial(epoch) / np.linalg.norm(self.compute_force_in_inertial(epoch))

        else:
            partial = np.zeros(3)

        return partial




def rotate_covariance(covariance, R):

    P = covariance
    full_R = np.zeros(covariance.shape)

    # block diagonal
    full_R[0:3, 0:3] = R
    full_R[3:6, 3:6] = R

    rotated_covariance = np.linalg.multi_dot([full_R, P, full_R.T])

    return rotated_covariance



# helper function for frame conversions of states and covariances
def smart_rotate(arr, R):

    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)
    elif len(arr.shape) == 2:
        pass
    else:
        raise RuntimeError(
            f"Array shape mismatch. Shape of input array {arr.shape} is not anticipated by rotation aux function.")

    if arr.shape[0] == 3:

        if arr.shape[1] == 1:

            return R.dot(arr)

        elif arr.shape[1] == arr.shape[1]:

            return np.linalg.multi_dot([R, arr, R.T])

        else:
            raise RuntimeError(f"Array shape mismatch. Shape of input array {arr.shape} is not anticipated by rotation aux function.")


    if arr.shape[0] == 6:

        large_R = np.zeros((6, 6))
        large_R[:3, :3] = R
        large_R[3:6, 3:6] = R

        if arr.shape[1] == 1:

            return large_R.dot(arr)

        elif arr.shape[1] == arr.shape[0]:

            return np.linalg.multi_dot([large_R, arr, large_R.T])

        else:
            raise RuntimeError(f"Array shape mismatch. Shape of input array {arr.shape} is not anticipated by rotation aux function.")

    else:
        raise RuntimeError(f"Array shape mismatch. Shape of input array {arr.shape} is not anticipated by rotation aux function.")


def main():


    ######################################################################################################
    ##### Create basic environment and simulation parameters
    ######################################################################################################


    # Set simulation start and end epochs
    simulation_start_epoch = DateTime(2024, 8, 28).epoch()
    simulation_end_epoch = simulation_start_epoch + 60*60*7

    # Load spice kernels
    spice.load_standard_kernels()

    # Create default body settings for "Sun", "Earth", "Moon"
    bodies_to_create = ["Sun", "Earth", "Moon"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation)


    ######################################################################################################
    ##### Create vehicle body and its environment interface
    ######################################################################################################

    # Create vehicle objects ("Starlink-32101", hereafter "Starlink")
    body_settings.add_empty_settings("StarLink")
    body_settings.get("StarLink").constant_mass = 260                       # [kg]

    # properties of starlink hall-effect thrusters
    thrust_magnitude = 170E-3                                               # [N]
    thrust_vector_body_fixed = np.array([0, 1, 0])                          # [-]
    thrust_magnitude_variance = (0.05 * thrust_magnitude)**2                # [N2]

    initial_state = np.loadtxt("starlink_initial_state.txt", delimiter=',') * 1E3                      # [m]
    initial_state_covariance_rsw = np.loadtxt("starlink_initial_covariance.txt", delimiter=',') * 1E6  # [m2]
    initial_state_covariance = smart_rotate(initial_state_covariance_rsw, frame_conversion.rsw_to_inertial_rotation_matrix(initial_state))



    ##### Aerodynamics Interface             #############################################################
    reference_area = 20                                                     # [m2] Average projection area of a 3U CubeSat
    drag_coefficient = 1.2                                                  # [-]
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0.0, 0.0]
    )
    body_settings.get("StarLink").aerodynamic_coefficient_settings = aero_coefficient_settings

    ##### Radiation Pressure Interface             #######################################################
    reference_area_radiation = 20  # Average projection area of a 3U CubeSat
    radiation_pressure_coefficient = 1.2
    occulting_bodies = dict()
    occulting_bodies["Sun"] = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
    )
    body_settings.get("StarLink").radiation_pressure_target_settings = radiation_pressure_settings


    """ HIGHLIGHT !!!
    ######################################################################################################
    ##### Define manoeuvre segments and associated sc orientation
    ######################################################################################################
    """

    # we have a 7h simulation period.
    # operational constraints require that we thrust in segments of 1h, with 2 hours between segments, resulting in 2 thrusting segments
    # manoeuvre design dictates that during first segment we thrust in the anti-n direction as defined by tnw frame convention
    # manoeuvre design dictates that during second segment we thrust in the n direction as defined by tnw frame convention

    ##### Define thrusting segments              #######################################################
    segments_epochs = (np.array([[2, 3], [5, 6]]) * (60*60) + simulation_start_epoch)

    segment_collection = []
    for i, seg_epochs in enumerate(segments_epochs):

        segment = ConstantThrustMagnitudeSegment(sc_name="StarLink", start_time=seg_epochs[0], end_time=seg_epochs[1])
        segment.set_thruster_properties(thrust_magnitude, thrust_vector_body_fixed, thrust_magnitude_variance)
        segment_collection.append(segment)

    multi_seg_manoeuvre = MyManoeuvre()
    multi_seg_manoeuvre.add_segments(segment_collection)


    ##### Define spacecraft orientation              ####################################################
    my_spacecraft_orientation = MySpacecraftOrientation(sc_name="StarLink")

    def roll_in_tnw(eval_epoch):  # we define a function that gives sc body fixed frame in tnw base frame.

        # at first, the rotation is eye (frames are aligned: x-->t, y-->n, z-->w),
        # after flip_epoch the sc has rotated 180 deg around its x axis (x-->t, y-->-n, z-->-w)

        flip_epoch = segments_epochs[1, 0] - 60  # flip orientation 60 s before onset of 2nd segment

        # no roll, we are orienting sc axis along tnw
        if eval_epoch < flip_epoch:
            return np.eye(3)
        # we roll 180 deg around t, such that y and z are pointing in anti n and anti w
        elif eval_epoch >= flip_epoch:
            R = Rotation.from_euler(seq='x', angles=180, degrees=True)
            return R.as_matrix()


    my_spacecraft_orientation.set_sc_rotation_ephemeris_in_tnw(roll_in_tnw)
    # --> self.inertial_to_body_fixed_callable
    # --> self.body_fixed_to_inertial_callable

    body_settings.get("StarLink").rotation_model_settings = environment_setup.rotation_model.custom_rotation_model(
        "J2000",
        "sc_fixed",
        custom_rotation_matrix_function=my_spacecraft_orientation.body_fixed_to_inertial_callable,
        finite_difference_time_step=20.)


    # Create system of bodies and link bodies object to my classes
    bodies = environment_setup.create_system_of_bodies(body_settings)
    my_spacecraft_orientation.set_bodies(bodies)
    multi_seg_manoeuvre.set_bodies(bodies)


    ######################################################################################################
    ##### Set Up Propagation
    ######################################################################################################

    # Define bodies that are propagated
    bodies_to_propagate = ["StarLink"]

    # Define central bodies of propagation
    central_bodies = ["Earth"]

    # Define the accelerations acting on `Starlink-32101`
    accelerations_settings_Starlink_32101 = dict(
        Sun=[
            propagation_setup.acceleration.radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Moon=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Earth=[
            propagation_setup.acceleration.spherical_harmonic_gravity(8, 8),
            propagation_setup.acceleration.aerodynamic()
        ],
        StarLink=[
            propagation_setup.acceleration.custom_acceleration(acceleration_function=multi_seg_manoeuvre.consolidated_custom_acceleration_callable)
        ]
    )

    # Create global accelerations dictionary
    acceleration_settings = {"StarLink": accelerations_settings_Starlink_32101}

    # Create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)


    # Create numerical integrator settings
    integrator_settings = propagation_setup.integrator. \
        runge_kutta_fixed_step_size(initial_time_step=60.0,
                                    coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)


    termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)


    dependent_variables_to_save = [
        propagation_setup.dependent_variable.single_acceleration(
            propagation_setup.acceleration.custom_acceleration_type, "StarLink", "StarLink"),
        propagation_setup.dependent_variable.inertial_to_body_fixed_313_euler_angles("StarLink")
    ]

    # Create propagation settings.
    translational_propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_condition,
        output_variables=dependent_variables_to_save
    )

    # This prints the initial and final state to ensure that the propagation is ran successfully, and has not terminated earlier.
    translational_propagator_settings.print_settings.print_initial_and_final_conditions = True


    ############## PARAMETERS ##########################

    """ HIGHLIGHT !!!
    ######################################################################################################
    ##### Define (CUSTOM) estimatable parameters
    ######################################################################################################
    """

    ### STANDARD: INITIAL STATE
    # Setup parameters settings to propagate the state transition matrix
    parameter_settings = estimation_setup.parameter.initial_states(translational_propagator_settings, bodies)

    # Add numerical partial of custom acceleration w.r.t. state
    parameter_settings[0].custom_partial_settings = [

        estimation_setup.parameter.custom_numerical_partial(

            parameter_perturbation=np.array([100, 100, 100, 1, 1, 1]),
            body_undergoing_acceleration="StarLink",
            body_exerting_acceleration="StarLink",
            acceleration_type=propagation_setup.acceleration.AvailableAcceleration.custom_acceleration_type

        )
    ]


    # (!!) define custom parameters, link getter/setter function and partials
    # custom parameter: thrust magnitude per segment
    for i, segment in enumerate(multi_seg_manoeuvre.segments):

        parameter_settings.append(
            estimation_setup.parameter.custom_parameter(
                f"thrust_magnitude_{i}", 1, segment.get_thrust_magnitude, segment.set_thrust_magnitude)
            )

        parameter_settings[-1].custom_partial_settings = [
            estimation_setup.parameter.custom_analytical_partial(
                segment.force_model_partial_to_thrust_magnitude, "StarLink", "StarLink",
                propagation_setup.acceleration.AvailableAcceleration.custom_acceleration_type
            )
        ]


    # Create the parameters that will be estimated
    parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)


    # Create simulation object and propagate dynamics
    variational_equations_solver = numerical_simulation.create_variational_equations_solver(
        bodies, translational_propagator_settings, parameters_to_estimate
    )


    # Retrieve all data produced by simulation
    dynamics_simulator = variational_equations_solver.dynamics_simulator

    # Extract the resulting state and dependent variable history and convert it to an ndarray
    states = dynamics_simulator.propagation_results.state_history
    states_array = result2array(states)
    epochs = states_array[:, 0]
    dep_vars = dynamics_simulator.propagation_results.dependent_variable_history
    dep_vars_array = result2array(dep_vars)
    custom_acc = dep_vars_array[:, 1:4]


    P_0 = linalg.block_diag(initial_state_covariance, np.diag(multi_seg_manoeuvre.get_thrust_magnitude_parameter_variance_vector()))

    covariance_history = dict()
    covariance_history_tnw = dict()
    formal_error_history = dict()
    formal_error_history_tnw = dict()

    custom_acc_in_tnw = np.zeros((len(epochs), 3))


    for i, epoch in enumerate(epochs):

        STM = variational_equations_solver.state_transition_matrix_history[epoch]
        S = variational_equations_solver.sensitivity_matrix_history[epoch]
        R_tnw = frame_conversion.inertial_to_tnw_rotation_matrix(states[epoch])

        P_t = np.linalg.multi_dot([np.hstack((STM, S)), P_0, np.hstack((STM, S)).transpose()])
        P_t_tnw = smart_rotate(P_t, R_tnw)

        covariance_history[epoch] = P_t
        covariance_history_tnw[epoch] = P_t_tnw

        formal_error_history[epoch] = np.sqrt(P_t.diagonal())
        formal_error_history_tnw[epoch] = np.sqrt(P_t_tnw.diagonal())

        # convert acceleration to tnw
        custom_acc_in_tnw[i, :] = smart_rotate(custom_acc[i, :], R_tnw).flatten()



    # Extract the resulting state and dependent variable history and convert it to an ndarray
    formal_error_history_array = result2array(formal_error_history)
    formal_error_history_tnw_array = result2array(formal_error_history_tnw)


    ######################################################################################################
    ##### Figures
    ######################################################################################################

    times_plot = (epochs-epochs[0]) / (3600)
    inertial_colors = ['#12223E', '#52B5C5', '#3274BC']
    tnw_colors = ['#662373', '#DC704D', '#F4BB47']


    ## Plot custom acceleration profile in tnw
    fig1, axs1 = plt.subplots(2)
    fig1.suptitle('Custom Acceleration (retrieved post-propagation)')

    axs1[0].set_title('... in inertial (J2000)')

    axs1[0].plot(times_plot, custom_acc[:, 0], label='x', color=inertial_colors[0])
    axs1[0].plot(times_plot, custom_acc[:, 1], label='y', color=inertial_colors[1])
    axs1[0].plot(times_plot, custom_acc[:, 2], label='z', color=inertial_colors[2])

    for seg in multi_seg_manoeuvre.segments:
        axs1[0].axvline((seg.start_time-epochs[0]) / (3600), color='grey', alpha=0.7, linestyle='--')
        axs1[0].axvline((seg.end_time-epochs[0]) / (3600), color='grey', alpha=0.7, linestyle='--')

    axs1[0].set_ylabel("acceleration [m/s2]")
    axs1[0].legend()

    axs1[1].set_title('... in tnw')

    axs1[1].plot(times_plot, custom_acc_in_tnw[:, 0], label='t', color=tnw_colors[0])
    axs1[1].plot(times_plot, custom_acc_in_tnw[:, 1], label='n', color=tnw_colors[1])
    axs1[1].plot(times_plot, custom_acc_in_tnw[:, 2], label='w', color=tnw_colors[2])

    for seg in multi_seg_manoeuvre.segments:
        axs1[1].axvline((seg.start_time-epochs[0]) / (3600), color='grey', alpha=0.7, linestyle='--')
        axs1[1].axvline((seg.end_time-epochs[0]) / (3600), color='grey', alpha=0.7, linestyle='--')

    axs1[1].legend()

    axs1[1].set_ylabel("acceleration [m/s2]")
    axs1[1].set_xlabel("Time (hours) since start")

    plt.tight_layout()
    plt.show()

    ## Plot custom acceleration profile in tnw
    fig2, axs2 = plt.subplots(2)
    fig2.suptitle('Covariance Evolution')

    axs2[0].set_title('... in inertial (J2000)')

    axs2[0].plot(times_plot, formal_error_history_array[:, 1], label='x', color=inertial_colors[0])
    axs2[0].plot(times_plot, formal_error_history_array[:, 2], label='y', color=inertial_colors[1])
    axs2[0].plot(times_plot, formal_error_history_array[:, 3], label='z', color=inertial_colors[2])

    for seg in multi_seg_manoeuvre.segments:
        axs2[0].axvline((seg.start_time-epochs[0]) / (3600), color='grey', alpha=0.7, linestyle='--')
        axs2[0].axvline((seg.end_time-epochs[0]) / (3600), color='grey', alpha=0.7, linestyle='--')


    axs2[0].set_ylabel("formal error [m]")
    axs2[0].legend()

    axs2[1].set_title('... in tnw')

    axs2[1].plot(times_plot, formal_error_history_tnw_array[:, 1], label='t', color=tnw_colors[0])
    axs2[1].plot(times_plot, formal_error_history_tnw_array[:, 2], label='n', color=tnw_colors[1])
    axs2[1].plot(times_plot, formal_error_history_tnw_array[:, 3], label='w', color=tnw_colors[2])

    for seg in multi_seg_manoeuvre.segments:
        axs2[1].axvline((seg.start_time-epochs[0]) / (3600), color='grey', alpha=0.7, linestyle='--')
        axs2[1].axvline((seg.end_time-epochs[0]) / (3600), color='grey', alpha=0.7, linestyle='--')

    axs2[1].legend()

    axs2[1].set_ylabel("formal error [m]")
    axs2[1].set_xlabel("Time (hours) since start")

    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    main()