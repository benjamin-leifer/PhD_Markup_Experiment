"""
PyBAMM integration for battery theoretical modeling.

This module provides interfaces to PyBAMM battery models, enabling simulation,
parameter estimation, and comparison with experimental data.
"""

import numpy as np
import logging
from functools import lru_cache
import json

try:
    import pybamm

    HAS_PYBAMM = True
except ImportError:
    HAS_PYBAMM = False
    logging.warning("PyBAMM not found. Install with: pip install pybamm")

try:
    from . import models, utils
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    models = importlib.import_module("models")
    utils = importlib.import_module("utils")

# Constants
DEFAULT_CHEMISTRY = "lithium-ion"
DEFAULT_MODEL = "SPM"  # Single Particle Model
AVAILABLE_MODELS = {
    "SPM": "Single Particle Model (faster, less detailed)",
    "DFN": "Doyle-Fuller-Newman Model (slower, more detailed)",
    "SPMe": "Single Particle Model with Electrolyte (balanced)",
    "Newman-Tobias": "Newman-Tobias Model (simplified electrolyte)",
    "Chen2020": "Chen et al. 2020 model with degradation",
}


def check_pybamm_availability():
    """Check if PyBAMM is available and return version if installed."""
    if not HAS_PYBAMM:
        return False, "PyBAMM not installed. Use 'pip install pybamm' to install."

    try:
        version = pybamm.__version__
        return True, f"PyBAMM version {version} installed."
    except Exception as e:
        return False, f"Error accessing PyBAMM: {str(e)}"


@lru_cache(maxsize=8)
def create_model(model_type=DEFAULT_MODEL, chemistry=DEFAULT_CHEMISTRY, options=None):
    """
    Create a PyBAMM model with specified type and chemistry.

    Args:
        model_type: Type of model (e.g. 'SPM', 'DFN')
        chemistry: Battery chemistry (e.g. 'lithium-ion', 'lead-acid')
        options: Dictionary of additional model options

    Returns:
        pybamm.BaseBatteryModel: The created model
    """
    if not HAS_PYBAMM:
        raise ImportError(
            "PyBAMM is required for theoretical modeling. Install with 'pip install pybamm'"
        )

    # Set default options if none provided
    if options is None:
        options = {}

    # Create the model based on model_type
    if model_type == "SPM":
        model = pybamm.lithium_ion.SPM(options=options)
    elif model_type == "DFN":
        model = pybamm.lithium_ion.DFN(options=options)
    elif model_type == "SPMe":
        model = pybamm.lithium_ion.SPMe(options=options)
    elif model_type == "Newman-Tobias":
        model = pybamm.lithium_ion.NewmanTobias(options=options)
    elif model_type == "Chen2020":
        # Model with degradation mechanisms
        model = pybamm.lithium_ion.Chen2020(options=options)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Available types: {list(AVAILABLE_MODELS.keys())}"
        )

    return model


def create_parameter_set(chemistry="NMC", options=None):
    """
    Create a parameter set for a given chemistry.

    Args:
        chemistry: Chemistry type shorthand ("NMC", "LFP", "NCA", "graphite", etc.)
        options: Additional parameter options

    Returns:
        dict: Parameter set
    """
    if not HAS_PYBAMM:
        raise ImportError(
            "PyBAMM is required for theoretical modeling. Install with 'pip install pybamm'"
        )

    # Map chemistry shorthand to PyBAMM parameter sets
    chemistry_map = {
        "NMC": "NMC_graphite",
        "LFP": "LFP_graphite",
        "NCA": "NCA_graphite",
        "LCO": "LCO_graphite",
        "LMO": "LMO_graphite",
    }

    pybamm_chemistry = chemistry_map.get(chemistry, chemistry)

    try:
        # Load the parameter set
        if options is not None:
            return pybamm.ParameterValues(chemistry=pybamm_chemistry, options=options)
        else:
            return pybamm.ParameterValues(chemistry=pybamm_chemistry)
    except Exception as e:
        logging.error(f"Error creating parameter set for {chemistry}: {e}")
        # Fall back to NMC_graphite if the specified chemistry fails
        return pybamm.ParameterValues(chemistry="NMC_graphite")


def extract_parameters_from_sample(sample_id):
    """
    Extract parameters from a Sample document to use in a PyBAMM model.

    Args:
        sample_id: ID of the sample to extract parameters from

    Returns:
        dict: Dictionary of parameters relevant to PyBAMM
    """
    # Retrieve the sample
    sample = models.Sample.objects(id=sample_id).first()
    if not sample:
        raise ValueError(f"Sample with ID {sample_id} not found")

    # Initialize the parameter dictionary
    parameters = {}

    # Extract chemistry information
    if hasattr(sample, "chemistry") and sample.chemistry:
        # Map chemistry string to model parameters
        chemistry_info = parse_chemistry_string(sample.chemistry)

        # Add to parameters
        parameters.update(chemistry_info)

    # Extract physical properties
    if hasattr(sample, "form_factor") and sample.form_factor:
        form_factor_params = extract_form_factor_parameters(sample.form_factor)
        parameters.update(form_factor_params)

    # Extract capacity information
    if hasattr(sample, "nominal_capacity") and sample.nominal_capacity:
        parameters["nominal_capacity"] = sample.nominal_capacity

    # Get Enhanced Sample properties if available
    if hasattr(sample, "components") and sample.components:
        component_params = extract_component_parameters(sample)
        parameters.update(component_params)

    # If testing data is available, extract useful information for validation
    if sample.tests:
        test_params = extract_test_parameters(sample.tests)
        parameters["validation_data"] = test_params

    return parameters


def parse_chemistry_string(chemistry_str):
    """
    Parse a chemistry string to extract cathode and anode materials.

    Args:
        chemistry_str: String description of chemistry (e.g. "LiNi0.8Mn0.1Co0.1O2/Graphite")

    Returns:
        dict: Dictionary with cathode and anode materials
    """
    params = {}

    # Common cathode materials to identify
    cathode_materials = {
        "NMC": ["NMC", "LiNiMnCoO2"],
        "NCA": ["NCA", "LiNiCoAlO2"],
        "LFP": ["LFP", "LiFePO4"],
        "LCO": ["LCO", "LiCoO2"],
        "LMO": ["LMO", "LiMn2O4"],
    }

    # Common anode materials
    anode_materials = {
        "graphite": ["graphite", "carbon", "C"],
        "silicon": ["silicon", "Si"],
        "LTO": ["LTO", "Li4Ti5O12"],
    }

    # Check for cathode material
    cathode_type = None
    for key, aliases in cathode_materials.items():
        if any(alias.lower() in chemistry_str.lower() for alias in aliases):
            cathode_type = key
            break

    if cathode_type:
        params["cathode_material"] = cathode_type

    # Check for anode material
    anode_type = None
    for key, aliases in anode_materials.items():
        if any(alias.lower() in chemistry_str.lower() for alias in aliases):
            anode_type = key
            break

    if anode_type:
        params["anode_material"] = anode_type

    # Look for stoichiometry in NMC (e.g., NMC811)
    if cathode_type == "NMC" and "NMC" in chemistry_str:
        # Look for numbers after NMC
        import re

        match = re.search(r"NMC(\d+)", chemistry_str)
        if match:
            digits = match.group(1)
            if len(digits) == 3:
                # Convert something like NMC811 to stoichiometry
                ni = int(digits[0]) / 10
                mn = int(digits[1]) / 10
                co = int(digits[2]) / 10
                params["cathode_stoichiometry"] = {"Ni": ni, "Mn": mn, "Co": co}

    return params


def chemistry_to_label(chemistry_str: str) -> str:
    """Return a formatted label like 'Cathode|Anode' from a chemistry string."""
    info = parse_chemistry_string(chemistry_str)
    cathode = info.get("cathode_material", "Cathode")
    anode = info.get("anode_material", "Anode")
    return f"{cathode}|{anode}"


def extract_form_factor_parameters(form_factor):
    """
    Extract geometric parameters based on form factor.

    Args:
        form_factor: String describing cell format (e.g. "18650", "pouch", "coin")

    Returns:
        dict: Dictionary with geometric parameters
    """
    params = {"form_factor": form_factor}

    # Cylindrical cells
    if form_factor == "18650":
        params["diameter"] = 18.6  # mm
        params["height"] = 65.2  # mm
        params["volume"] = 17.7  # cm³ (calculated from dimensions)
        params["geometry"] = "cylindrical"

    elif form_factor == "21700":
        params["diameter"] = 21.0  # mm
        params["height"] = 70.0  # mm
        params["volume"] = 24.2  # cm³
        params["geometry"] = "cylindrical"

    # Pouch cell (approximate)
    elif "pouch" in form_factor.lower():
        params["geometry"] = "pouch"

    # Coin cell (typical 2032)
    elif "coin" in form_factor.lower() or form_factor == "2032":
        params["diameter"] = 20.0  # mm
        params["height"] = 3.2  # mm
        params["volume"] = 1.0  # cm³
        params["geometry"] = "coin"

    return params


def extract_component_parameters(sample):
    """
    Extract parameters from a sample's components.

    Args:
        sample: EnhancedSample with component information

    Returns:
        dict: Dictionary with component parameters
    """
    params = {}

    # Check if sample has components attribute
    if not hasattr(sample, "components") or not sample.components:
        return params

    # Process each component
    for component in sample.components:
        role = component.role_type
        material = component.material

        if role == "cathode":
            params["cathode"] = {
                "material": getattr(material, "name", "unknown"),
                "loading": (
                    component.properties.get("loading", {}).get("value", None)
                    if hasattr(component, "properties")
                    else None
                ),
            }

        elif role == "anode":
            params["anode"] = {
                "material": getattr(material, "name", "unknown"),
                "loading": (
                    component.properties.get("loading", {}).get("value", None)
                    if hasattr(component, "properties")
                    else None
                ),
            }

        elif role == "electrolyte":
            params["electrolyte"] = {"material": getattr(material, "name", "unknown")}

    return params


def extract_test_parameters(test_refs):
    """
    Extract useful parameters from test results for model validation.

    Args:
        test_refs: List of test references

    Returns:
        dict: Dictionary with test parameters
    """
    test_params = []

    for test_ref in test_refs:
        test = models.TestResult.objects(id=test_ref.id).first()
        if not test:
            continue

        # Extract test conditions
        test_data = {
            "test_id": str(test.id),
            "test_name": test.name,
            "cycle_count": getattr(test, "cycle_count", 0),
        }

        # Extract temperature if available
        if hasattr(test, "temperature") and test.temperature is not None:
            test_data["temperature"] = test.temperature

        # Extract voltage limits
        if (
            hasattr(test, "upper_cutoff_voltage")
            and test.upper_cutoff_voltage is not None
        ):
            test_data["upper_voltage"] = test.upper_cutoff_voltage

        if (
            hasattr(test, "lower_cutoff_voltage")
            and test.lower_cutoff_voltage is not None
        ):
            test_data["lower_voltage"] = test.lower_cutoff_voltage

        # Extract C-rates if available
        if hasattr(test, "charge_rate") and test.charge_rate is not None:
            test_data["charge_C_rate"] = test.charge_rate

        if hasattr(test, "discharge_rate") and test.discharge_rate is not None:
            test_data["discharge_C_rate"] = test.discharge_rate

        test_params.append(test_data)

    return test_params


def run_simulation(
    model_type=DEFAULT_MODEL, parameters=None, experiment=None, solver=None
):
    """
    Run a PyBAMM simulation with the given model and parameters.

    Args:
        model_type: Type of model to simulate
        parameters: Dictionary of model parameters or PyBAMM ParameterValues object
        experiment: PyBAMM Experiment to simulate (or parameters to create one)
        solver: Optional custom solver to use

    Returns:
        dict: Simulation results
    """
    if not HAS_PYBAMM:
        raise ImportError(
            "PyBAMM is required for theoretical modeling. Install with 'pip install pybamm'"
        )

    # Create model
    model = create_model(model_type)

    # Process parameters
    if parameters is None:
        # Use default parameters
        param = pybamm.ParameterValues(chemistry=DEFAULT_CHEMISTRY)
    elif isinstance(parameters, dict):
        # Convert dictionary to PyBAMM parameters
        param = parameters_dict_to_pybamm(parameters)
    else:
        # Assume it's already a PyBAMM ParameterValues object
        param = parameters

    # Process experiment
    if experiment is None:
        # Default to 1C discharge
        experiment = pybamm.Experiment(
            [
                "Discharge at 1C until 2.5 V",
            ]
        )
    elif isinstance(experiment, dict):
        # Create experiment from dictionary
        experiment = create_experiment_from_dict(experiment)

    # Create the solver
    if solver is None:
        solver = pybamm.CasadiSolver()

    # Set up and run simulation
    sim = pybamm.Simulation(
        model, experiment=experiment, solver=solver, parameter_values=param
    )

    try:
        # Run the simulation
        sim.solve()

        # Process results
        results = process_simulation_results(sim)
        return results

    except Exception as e:
        logging.error(f"Simulation error: {str(e)}")
        raise RuntimeError(f"Simulation failed: {str(e)}")


def parameters_dict_to_pybamm(parameters):
    """
    Convert a parameters dictionary to a PyBAMM ParameterValues object.

    Args:
        parameters: Dictionary of parameters

    Returns:
        pybamm.ParameterValues: Parameters in PyBAMM format
    """
    if not HAS_PYBAMM:
        raise ImportError(
            "PyBAMM is required for theoretical modeling. Install with 'pip install pybamm'"
        )

    # Start with a default parameter set
    chemistry = parameters.get("chemistry", "NMC_graphite")

    if isinstance(chemistry, str):
        # Map common abbreviations to PyBAMM parameter sets
        chemistry_map = {
            "NMC": "NMC_graphite",
            "LFP": "LFP_graphite",
            "NCA": "NCA_graphite",
            "LCO": "LCO_graphite",
        }
        chemistry = chemistry_map.get(chemistry, chemistry)

    # Create parameter values from the base chemistry
    param = pybamm.ParameterValues(chemistry=chemistry)

    # Update with custom parameters if provided
    custom_parameters = parameters.get("custom_parameters", {})
    if custom_parameters:
        param.update(custom_parameters)

    return param


def create_experiment_from_dict(experiment_dict):
    """
    Create a PyBAMM Experiment object from a dictionary.

    Args:
        experiment_dict: Dictionary with experiment specifications

    Returns:
        pybamm.Experiment: The created experiment
    """
    if not HAS_PYBAMM:
        raise ImportError(
            "PyBAMM is required for theoretical modeling. Install with 'pip install pybamm'"
        )

    # Extract period specifications
    period_specs = experiment_dict.get("period_specs", [])

    if not period_specs:
        # Create a default experiment if none specified
        return pybamm.Experiment(["Discharge at 1C until 2.5 V"])

    # Convert each period spec to a PyBAMM string format
    period_strs = []

    for spec in period_specs:
        period_type = spec.get("type", "")

        if period_type == "CC_charge":
            rate = spec.get("rate", 1)
            cutoff_voltage = spec.get("cutoff_voltage", 4.2)
            period_strs.append(f"Charge at {rate}C until {cutoff_voltage} V")

        elif period_type == "CC_discharge":
            rate = spec.get("rate", 1)
            cutoff_voltage = spec.get("cutoff_voltage", 2.5)
            period_strs.append(f"Discharge at {rate}C until {cutoff_voltage} V")

        elif period_type == "CCCV_charge":
            c_rate = spec.get("c_rate", 1)
            v_hold = spec.get("v_hold", 4.2)
            c_cutoff = spec.get("c_cutoff", 0.05)
            period_strs.append(f"Charge at {c_rate}C until {v_hold} V")
            period_strs.append(f"Hold at {v_hold} V until {c_cutoff}C")

        elif period_type == "rest":
            time = spec.get("time", 30)
            period_strs.append(f"Rest for {time} minutes")

    # Create the experiment
    return pybamm.Experiment(period_strs)


def process_simulation_results(simulation):
    """
    Process PyBAMM simulation results into a more usable format.

    Args:
        simulation: PyBAMM Simulation object after solving

    Returns:
        dict: Processed simulation results
    """
    if not hasattr(simulation, "solution") or simulation.solution is None:
        return {"error": "No solution available"}

    solution = simulation.solution
    model = simulation.model

    # Create a results dictionary
    results = {
        "model": model.name,
        "has_solution": True,
        "times": solution.t,
        "cycles": {},
    }

    # Extract important variables for each cycle
    for cycle_num, cycle_solution in enumerate(solution.cycles, start=1):
        # Get time vector for this cycle
        t = cycle_solution.t

        # Extract variables if they exist in the solution
        variables = {}

        # Common variables to extract
        var_names = [
            "Terminal voltage [V]",
            "Current [A]",
            "Discharge capacity [A.h]",
            "Negative particle surface concentration",
            "Positive particle surface concentration",
            "Cell temperature [K]",
            "X-averaged electrolyte concentration [mol.m-3]",
        ]

        for var_name in var_names:
            try:
                if var_name in cycle_solution.all_ys_dict:
                    variables[var_name] = cycle_solution[var_name].data.tolist()
            except Exception:
                # Skip variables that can't be extracted
                pass

        # Add to results dictionary
        results["cycles"][cycle_num] = {"t": t.tolist(), "variables": variables}

    return results


def compare_simulation_with_experiment(simulation_results, test_id):
    """
    Compare simulation results with experimental data from a test.

    Args:
        simulation_results: Results from run_simulation
        test_id: ID of the TestResult to compare with

    Returns:
        dict: Comparison metrics and data
    """
    # Get the experimental data
    test = models.TestResult.objects(id=test_id).first()
    if not test:
        raise ValueError(f"Test with ID {test_id} not found")

    # Extract cycle data from the test
    try:
        from . import analysis
    except ImportError:  # pragma: no cover - allow running as script
        import importlib

        analysis = importlib.import_module("analysis")
    cycle_data = analysis.get_cycle_data(test_id)

    # Prepare comparison results
    comparison = {
        "test_name": test.name,
        "test_id": str(test.id),
        "sample_name": utils.get_sample_name(test.sample),
        "model": simulation_results.get("model", "Unknown"),
        "metrics": {},
        "cycles": {},
    }

    # Compare each cycle
    for cycle_num, sim_cycle in simulation_results.get("cycles", {}).items():
        # Find matching experimental cycle
        exp_cycle = None
        for c in cycle_data.get("cycles", []):
            if c["cycle_index"] == cycle_num:
                exp_cycle = c
                break

        if exp_cycle is None:
            continue

        # Compare capacity
        sim_capacity = None
        if "Discharge capacity [A.h]" in sim_cycle["variables"]:
            # Get final capacity value
            sim_capacity = (
                sim_cycle["variables"]["Discharge capacity [A.h]"][-1] * 1000
            )  # Convert A.h to mAh

        exp_capacity = exp_cycle["discharge_capacity"]

        # Calculate error
        if sim_capacity is not None:
            capacity_error = (
                (sim_capacity - exp_capacity) / exp_capacity * 100
                if exp_capacity > 0
                else float("inf")
            )

            # Add to comparison
            comparison["cycles"][cycle_num] = {
                "sim_capacity": sim_capacity,
                "exp_capacity": exp_capacity,
                "capacity_error_pct": capacity_error,
            }

    # Calculate overall metrics
    if comparison["cycles"]:
        capacity_errors = [
            c["capacity_error_pct"] for c in comparison["cycles"].values()
        ]

        comparison["metrics"]["mean_capacity_error_pct"] = np.mean(capacity_errors)
        comparison["metrics"]["max_capacity_error_pct"] = np.max(capacity_errors)
        comparison["metrics"]["min_capacity_error_pct"] = np.min(capacity_errors)
        comparison["metrics"]["rmse_capacity"] = np.sqrt(
            np.mean(np.array(capacity_errors) ** 2)
        )

    return comparison


def estimate_model_parameters(
    test_id, model_type=DEFAULT_MODEL, initial_params=None, params_to_fit=None
):
    """
    Estimate model parameters by fitting to experimental data.

    Args:
        test_id: ID of the TestResult to fit to
        model_type: Type of model to use
        initial_params: Initial parameter guess
        params_to_fit: List of parameter names to fit

    Returns:
        dict: Fitted parameters and goodness of fit metrics
    """
    if not HAS_PYBAMM:
        raise ImportError(
            "PyBAMM is required for parameter estimation. Install with 'pip install pybamm'"
        )

    try:
        import pybamm.parameter_inference as parameterization
    except ImportError:
        raise ImportError(
            "Could not import parameter inference module. Make sure you have a recent version of PyBAMM."
        )

    # Get the experimental data
    test = models.TestResult.objects(id=test_id).first()
    if not test:
        raise ValueError(f"Test with ID {test_id} not found")

    # Extract data for fitting
    exp_data = extract_pybamm_experiment_data(test)

    # Set default parameters to fit if not specified
    if params_to_fit is None:
        params_to_fit = [
            "Negative electrode diffusivity [m2.s-1]",
            "Positive electrode diffusivity [m2.s-1]",
        ]

    # Create model
    model = create_model(model_type)

    # Set up parameter fitting
    problem = parameterization.SingleModelProblem(model, exp_data)

    # Run parameter fitting
    result = problem.fit(
        initial_guess=initial_params, parameters=params_to_fit, bounds=None
    )

    # Extract fitted parameters
    fitted_params = {
        param: float(value) for param, value in zip(params_to_fit, result.x)
    }

    # Run a simulation with the fitted parameters
    param_values = pybamm.ParameterValues(chemistry=DEFAULT_CHEMISTRY)
    param_values.update({p: v for p, v in fitted_params.items()})

    sim = pybamm.Simulation(model)
    sim.solve(param_values)

    # Calculate goodness of fit
    goodness_of_fit = calculate_goodness_of_fit(sim.solution, exp_data)

    return {
        "fitted_parameters": fitted_params,
        "goodness_of_fit": goodness_of_fit,
        "initial_params": initial_params,
        "iteration_count": result.nit,
    }


def extract_pybamm_experiment_data(test):
    """
    Extract data from a TestResult in a format suitable for PyBAMM.

    Args:
        test: TestResult document

    Returns:
        dict: Data formatted for PyBAMM
    """
    data = {}

    # Check for cycles
    if not test.cycles:
        return data

    # Extract time-series data from cycles
    cycle_data = []

    for cycle in test.cycles:
        # We'd need time-series data within cycles, which isn't stored in our standard model
        # This is a simplified placeholder assuming we can get the time data
        time_data = [0, 3600]  # Placeholder, one hour per cycle
        current_data = [1.0, 1.0]  # Placeholder, 1C discharge
        voltage_data = [4.2, 3.0]  # Placeholder, typical voltage range

        cycle_data.append(
            {
                "cycle": cycle.cycle_index,
                "time": time_data,
                "current": current_data,
                "voltage": voltage_data,
                "temperature": [298.15, 298.15],  # Placeholder, ambient temperature
                "capacity": cycle.discharge_capacity / 1000.0,  # Convert mAh to Ah
            }
        )

    # Format for PyBAMM
    data["cycles"] = cycle_data

    return data


def calculate_goodness_of_fit(solution, experimental_data):
    """
    Calculate goodness of fit metrics between simulation and experimental data.

    Args:
        solution: PyBAMM solution
        experimental_data: Experimental data used for fitting

    Returns:
        dict: Goodness of fit metrics
    """
    # Placeholder for goodness of fit calculation
    # In a full implementation, this would interpolate the solution to the experimental data points
    # and calculate metrics like RMSE, R², etc.

    return {"rmse_voltage": 0.05, "r_squared": 0.95}  # Placeholder values


def predict_cycle_life(
    model_type="Chen2020", parameters=None, operating_conditions=None
):
    """
    Predict cycle life using a degradation model.

    Args:
        model_type: Type of model to use (should include degradation physics)
        parameters: Model parameters
        operating_conditions: Dictionary of operating conditions

    Returns:
        dict: Prediction results including cycle life and degradation metrics
    """
    if not HAS_PYBAMM:
        raise ImportError(
            "PyBAMM is required for cycle life prediction. Install with 'pip install pybamm'"
        )

    # Set default operating conditions
    if operating_conditions is None:
        operating_conditions = {
            "temperature": 298.15,  # K (25°C)
            "upper_voltage": 4.2,  # V
            "lower_voltage": 2.5,  # V
            "charge_crate": 0.5,  # C
            "discharge_crate": 1.0,  # C
            "cycles": 500,  # Number of cycles to simulate
        }

    # Create a degradation model
    model = create_model(model_type)

    # Set up experiment to match operating conditions
    experiment = pybamm.Experiment(
        [
            f"Charge at {operating_conditions['charge_crate']}C until {operating_conditions['upper_voltage']} V",
            f"Discharge at {operating_conditions['discharge_crate']}C until {operating_conditions['lower_voltage']} V",
        ]
        * operating_conditions["cycles"]
    )

    # Process parameters
    if parameters is None:
        param = pybamm.ParameterValues(chemistry="NMC_graphite")
    elif isinstance(parameters, dict):
        param = parameters_dict_to_pybamm(parameters)
    else:
        param = parameters

    # Update temperature
    param.update({"Ambient temperature [K]": operating_conditions["temperature"]})

    # Run the simulation
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
    sol = sim.solve()

    # Extract degradation metrics
    # For capacity fade over cycles
    cycle_numbers = []
    capacities = []
    for i, cycle_sol in enumerate(sol.cycles):
        if i % 10 == 0:  # Sample every 10 cycles to reduce data size
            cycle_numbers.append(i + 1)
            idx = cycle_sol.cycle_steps[-1][0]  # Last step of the cycle
            capacities.append(cycle_sol["Discharge capacity [A.h]"](idx))

    cycle_numbers = np.array(cycle_numbers)
    capacities = np.array(capacities)

    # Calculate capacity fade
    initial_capacity = capacities[0]
    capacity_fade_pct = (initial_capacity - capacities) / initial_capacity * 100

    # Estimate cycle life to 80% capacity
    try:
        from scipy import interpolate

        if np.min(capacity_fade_pct) > 20:
            # Reached 80% capacity, interpolate to find exact cycle
            f = interpolate.interp1d(capacity_fade_pct, cycle_numbers)
            cycle_life_80 = float(f(20))
        else:
            # Extrapolate to estimate cycle life
            from scipy import optimize

            # Fit a curve to the fade data
            def power_law(x, a, b, c):
                return a * np.power(x, b) + c

            params, _ = optimize.curve_fit(
                power_law,
                cycle_numbers,
                capacity_fade_pct,
                p0=[0.1, 0.5, 0],
                maxfev=10000,
            )

            # Find when the fitted curve reaches 20% fade
            def find_eol(x):
                return power_law(x, *params) - 20

            try:
                cycle_life_80 = float(
                    optimize.brentq(find_eol, cycle_numbers[-1], cycle_numbers[-1] * 10)
                )
            except Exception:
                cycle_life_80 = float("inf")  # Could not determine EOL
    except Exception as e:
        logging.error(f"Error estimating cycle life: {str(e)}")
        cycle_life_80 = None

    # Prepare results
    results = {
        "model": model_type,
        "operating_conditions": operating_conditions,
        "initial_capacity_ah": float(initial_capacity),
        "final_capacity_ah": float(capacities[-1]),
        "capacity_fade_pct": float(capacity_fade_pct[-1]),
        "cycle_life_80pct": cycle_life_80,
        "simulated_cycles": operating_conditions["cycles"],
        "data": {
            "cycle_numbers": cycle_numbers.tolist(),
            "capacities_ah": capacities.tolist(),
            "capacity_fade_pct": capacity_fade_pct.tolist(),
        },
    }

    return results


def save_model_parameters(parameters, filepath):
    """
    Save model parameters to a JSON file.

    Args:
        parameters: Dictionary of parameters
        filepath: Path to save the file

    Returns:
        bool: True if successful
    """
    # Convert parameters to a serializable format
    serializable_params = {}

    for key, value in parameters.items():
        if isinstance(value, (int, float, str, bool, list, dict)) or value is None:
            serializable_params[key] = value
        elif hasattr(value, "__dict__"):
            # Convert objects to dictionaries
            serializable_params[key] = value.__dict__
        else:
            # Convert anything else to string representation
            serializable_params[key] = str(value)

    # Save to file
    try:
        with open(filepath, "w") as f:
            json.dump(serializable_params, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving parameters: {str(e)}")
        return False


def load_model_parameters(filepath):
    """
    Load model parameters from a JSON file.

    Args:
        filepath: Path to the parameter file

    Returns:
        dict: Loaded parameters
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading parameters: {str(e)}")
        return {}
