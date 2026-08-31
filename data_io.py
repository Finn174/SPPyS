"""
data_io.py

Input/output routines for SPPyS.

Handles

- JSON parameter files for input of initial parameters and output of fitted parameters
- DAT data files for RSMs to be fitted to
- CSV data files for RSMs to be fitted to and forward simulations to be saved as
"""

import json
import numpy as np
import pandas as pd

from parameters import SimulationParameters


# --------------------------------------------------------
# JSON
# --------------------------------------------------------

def load_parameters(filename):
    """
    Loads a JSON parameter file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    SimulationParameters object
    """

    with open(filename, "r") as f:
        p = json.load(f)
    
    # Convert lists to numpy arrays
    Qin = np.array(p["Qin"], dtype=float)
    Qout = np.array(p["Qout"], dtype=float)
    Bragg_peak = np.array(p["Bragg_peak"], dtype=float)
    prop_vector = np.array(p["prop_vector"], dtype=float)

    # Optional substrate
    Sub_peak = p.get("Sub_peak")

    if Sub_peak is not None:
        Sub_peak = np.array(Sub_peak, dtype=float)

    # Allow either delta or Lambda + a
    delta = p.get("delta")
    if delta is None:
        Lambda = p.get("Lambda")
        a = p.get("a")
        if Lambda is None or a is None:
            raise ValueError("Provide delta OR (Lambda and a).")
        delta = a / Lambda

    params = SimulationParameters(

        Qin=Qin,
        Qout=Qout,

        Bragg_peak=Bragg_peak,
        prop_vector=prop_vector,

        delta=delta,
        delta_bounds = p.get("delta_bounds", [None, None]),

        background=p["background"],
        background_bounds = p.get("background_bounds", [None, None]),

        film_intensity=p["film_intensity"],
        film_intensity_bounds = p.get("film_intensity_bounds", [None, None]),

        sub_intensity=p.get("sub_intensity", 0),
        sub_intensity_bounds = p.get("sub_intensity_bounds", [None, None]),

        film_long=p["FWHM_film_long"],
        film_long_bounds = p.get("film_long_bounds", [None, None]),
        
        film_short=p["FWHM_film_short"],
        film_short_bounds = p.get("film_short_bounds", [None, None]),

        substrate_long=p.get("FWHM_s_long", 0.03),
        substrate_long_bounds = p.get("substrate_long_bounds", [None, None]),

        substrate_short=p.get("FWHM_s_short", 0.01),
        substrate_short_bounds = p.get("substrate_short_bounds", [None, None]),

        sgl_width=p["sgl_width"],

        In_Range=p.get("In_Range", (0.48, 0.51)),

        Out_Range=p.get("Out_Range", (0.48, 0.51)),

        shift_x=p.get("shift_x", 0),
        shift_x_bounds = p.get("shift_x_bounds", [None, None]),

        shift_y=p.get("shift_y", 0),
        shift_y_bounds = p.get("shift_y_bounds", [None, None]),

        shift_angle=p.get("shift_angle", 0),
        shift_angle_bounds = p.get("shift_angle_bounds", [None, None]),

        shift_x_sub=p.get("shift_x_sub", 0),
        shift_x_sub_bounds = p.get("shift_x_sub_bounds", [None, None]),

        shift_y_sub=p.get("shift_y_sub", 0),
        shift_y_sub_bounds = p.get("shift_y_sub_bounds", [None, None]),

        shift_angle_sub=p.get("shift_angle_sub", 0),
        shift_angle_sub_bounds = p.get("shift_angle_sub_bounds", [None, None]),

        Sub_peak=Sub_peak,

        axis_compress=p.get("axis_compress", 1),

        xlabel=p.get("xlabel", None),

        ylabel=p.get("ylabel", None),
    )

    return params


# --------------------------------------------------------
# Save parameters
# --------------------------------------------------------

def save_parameters(filename, params):
    """
    Saves a SimulationParameters object to JSON.
    """

    Bragg_peak_fit = params.Bragg_peak + params.Qin * params.shift_x + params.Qout * params.shift_y

    if params.Sub_peak is not None:
        Sub_peak_fit = params.Sub_peak + params.Qin * params.shift_x_sub + params.Qout * params.shift_y_sub
    else:
        Sub_peak_fit = None

    output = {

        "Qin": params.Qin.tolist(),

        "Qout": params.Qout.tolist(),

        "Bragg_peak": Bragg_peak_fit.tolist(),

        "prop_vector": params.prop_vector.tolist(),

        "delta": params.delta,

        "background": params.background,

        "film_intensity": params.film_intensity,

        "sub_intensity": params.sub_intensity,

        "FWHM_film_long": params.film_long,

        "FWHM_film_short": params.film_short,

        "FWHM_s_long": params.substrate_long,

        "FWHM_s_short": params.substrate_short,

        "sgl_width": params.sgl_width,

        "In_Range": list(params.In_Range),

        "Out_Range": list(params.Out_Range),

        "shift_x": params.shift_x,

        "shift_y": params.shift_y,

        "shift_angle": params.shift_angle,
        
        "Sub_peak": Sub_peak_fit.tolist() if Sub_peak_fit is not None else None,

        "shift_x_sub": params.shift_x_sub if Sub_peak_fit is not None else None,

        "shift_y_sub": params.shift_y_sub if Sub_peak_fit is not None else None,

        "shift_angle_sub": params.shift_angle_sub if Sub_peak_fit is not None else None,

        "axis_compress": params.axis_compress,

        "xlabel": params.xlabel,

        "ylabel": params.ylabel,
    }

    with open(filename, "w") as f:

        json.dump(
            output,
            f,
            indent=4,
        )


# --------------------------------------------------------
# Experimental data
# --------------------------------------------------------

def cluster_coordinates(values, tolerance):
    """
    Clusters nearly identical coordinate values.

    Parameters
    ----------
    values : array-like
        Coordinate values.

    tolerance : float
        Maximum separation between values belonging
        to the same coordinate.

    Returns
    -------
    centres
        Sorted cluster centres.

    mapped
        Original values mapped onto the cluster centres.
    """

    values = np.asarray(values)

    unique = np.sort(values)

    clusters = [[unique[0]]]

    for value in unique[1:]:

        if abs(value - clusters[-1][-1]) <= tolerance:

            clusters[-1].append(value)

        else:

            clusters.append([value])

    centres = np.array([
        np.mean(cluster)
        for cluster in clusters
    ])

    mapped = np.empty_like(values)

    for i, value in enumerate(values):

        index = np.argmin(np.abs(centres - value))

        mapped[i] = centres[index]

    return centres, mapped


def map_HKL_to_XY(H, K, L, Qin, Qout):
    """
    Maps reciprocal-space (H, K, L) coordinates to scattering-plane (X, Y).

    Uses the provided Qin and Qout vectors to determine the projection.

    Parameters
    ----------
    H, K, L : ndarray
        Reciprocal-space coordinates from data file

    Qin : ndarray
        In-plane Q-vector (shape (3,))

    Qout : ndarray
        Out-of-plane Q-vector (shape (3,))

    Returns
    -------
    X, Y : ndarray
        Projected coordinates in the scattering plane
    """

    # Build plane basis
    n = np.cross(Qin, Qout)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        raise ValueError("Qin and Qout are parallel - cannot define scattering plane")
    n /= n_norm

    u = Qin - np.dot(Qin, n) * n
    u /= np.linalg.norm(u)

    v = np.cross(n, u)

    # Assume cubic lattice: reciprocal lattice vectors are along [100], [010], [001]
    # Map: H → [1,0,0], K → [0,1,0], L → [0,0,1]
    HKL_vectors = np.array([
        [1, 0, 0],  # H direction
        [0, 1, 0],  # K direction
        [0, 0, 1],  # L direction
    ])

    # Project each reciprocal lattice direction onto the scattering plane
    projections = np.array([
        [np.dot(HKL_vectors[i], u), np.dot(HKL_vectors[i], v)]
        for i in range(3)
    ])

    # Find which has the largest projection in u direction (in-plane)
    u_projections = np.abs(projections[:, 0])
    in_plane_idx = np.argmax(u_projections)

    # Find which has the largest projection in v direction (out-of-plane)
    v_projections = np.abs(projections[:, 1])
    out_of_plane_idx = np.argmax(v_projections)

    # Handle case where same index is picked twice
    remaining_indices = [i for i in range(3) if i not in [in_plane_idx, out_of_plane_idx]]
    if not remaining_indices:
        # Same index picked twice - use second largest
        v_sorted = np.argsort(v_projections)[::-1]
        for idx in v_sorted:
            if idx != in_plane_idx:
                out_of_plane_idx = idx
                break

    labels = ['H', 'K', 'L']
    in_plane_label = labels[in_plane_idx]
    out_of_plane_label = labels[out_of_plane_idx]

    print(f"Detected scan type: {in_plane_label}{out_of_plane_label}")
    print(f"  In-plane ({in_plane_label}): projection on u = {projections[in_plane_idx, 0]:.4f}")
    print(f"  Out-of-plane ({out_of_plane_label}): projection on v = {projections[out_of_plane_idx, 1]:.4f}")

    # Extract X and Y based on detected indices
    data_array = np.array([H, K, L])
    X = data_array[in_plane_idx]
    Y = data_array[out_of_plane_idx]

    return X, Y


def parse_column_headers(headers):
    """
    Parses column headers to identify reciprocal-space directions and intensity.

    Works for DAT files with ordering Q_H, Q_K, Q_L, Intensity or Qin, Qout, Intensity.
    
    Handles various naming conventions like:
    - 'H', 'K', 'L' (single indices)
    - 'Q_H', 'Q_K', 'Q_L' (prefixed)
    - 'Q_H-K', 'Q_HK', 'Q_2H' (linear combinations)
    - Case insensitive
    
    Returns dict with:
        'indices': list of (h_coeff, k_coeff, l_coeff) tuples for each data column
        'intensity_ind': index of intensity column
        'labels': labels for each direction
    or None if parsing fails.
    """
    headers_lower = [h.lower() for h in headers]
    
    indices_list = []
    intensity_ind = None
    labels = []
    
    for i, header in enumerate(headers_lower):
        # Remove 'q' and '_' characters
        normalised = header.replace('q', '').replace('_', '').lower()
        
        # Check if column is intensity
        if 'intensity' in normalised or normalised == 'z':
            intensity_ind = i
        
        # Try to parse Miller indices (H, K, L coefficients)
        elif intensity_ind is None:  # Only parse until reaching intensity
            h_coeff, k_coeff, l_coeff = parse_miller_combination(normalised)
        
            if h_coeff is not None:  # Successfully parsed
                indices_list.append((h_coeff, k_coeff, l_coeff))
                labels.append(normalised)
    
    # If intensity not found, assume last column
    if intensity_ind is None:
        intensity_ind = len(headers) - 1
    
    # Check if enough reciprocal-space columns found
    if len(indices_list) < 2:
        return None
    
    return {
        'indices': indices_list,
        'intensity_ind': intensity_ind,
        'labels': labels,
    }


def parse_miller_combination(label):
    """
    Parses Miller index combinations like 'H', 'K', 'H-K', 'HK', '2HL', etc.
    
    Returns (h_coeff, k_coeff, l_coeff) or (None, None, None) if parsing fails.
    """
    label = label.strip().lower()
    
    # Initialize coefficients
    coeffs = {'h': 0, 'k': 0, 'l': 0}
    
    chars = list(label)

    sign = 1
    coeff = 1
    index = 0
    indices = ['h', 'k', 'l']
    buffer = ''

    if len(chars) == 1:
        if chars[0] == 'h':
            return 1, 0, 0
        elif chars[0] == 'k':
            return 0, 1, 0
        elif chars[0] == 'l':
            return 0, 0, 1
    
    if len(chars) == 2 and chars[0] == '-':
        if chars[1] == 'h':
            return -1, 0, 0
        elif chars[1] == 'k':
            return 0, -1, 0
        elif chars[1] == 'l':
            return 0, 0, -1

    for char in chars:
        if char == '-':
            sign = -1
        
        elif char.isdigit() and char != '0':
            if buffer == '0.' or buffer == '0':
                index += 1
                buffer = ''
                sign = 1
            else:
                # Handle coefficients like '2H'
                coeff = int(char) * sign
                sign = 1  # Reset sign after using it

        elif char == '0':
            buffer = '0'
        
        elif char == '.' and buffer == '0':
            # Handle decimal point after zero
            buffer += '.'
        
        else:
            if buffer == '0':
                index += 1
                buffer = ''
            else:
                coeffs[indices[index]] += coeff * sign
                sign = 1  # Reset sign after using it
                index += 1
                coeff = 1  # Reset coefficient after using it
    
    # If no indices were parsed, return None
    if sum(abs(v) for v in coeffs.values()) == 0:
        return None, None, None
    
    return coeffs['h'], coeffs['k'], coeffs['l']


def load_experimental_data(filename, 
                          coordinate_tolerance=5e-4,
                          Qin=None,
                          Qout=None):
    """
    Loads an experimental reciprocal-space map.

    Supported formats
    -----------------
    CSV:
        X, Y, Z

    DAT:
        H, K, L, Intensity
        Qin, Qout, Intensity

    Parameters
    ----------
    filename : str
        Data file path
    
    coordinate_tolerance : float
        Tolerance for clustering coordinates
    
    Qin : ndarray, optional
        In-plane Q-vector
    
    Qout : ndarray, optional
        Out-of-plane Q-vector
    
    Returns
    -------
    xs, ys, image
    """

    extension = filename.split(".")[-1].lower()

    # ======================================================
    # CSV
    # ======================================================

    if extension == "csv":
        df = pd.read_csv(filename)
        x = np.sort(df["X"].unique())
        y = np.sort(df["Y"].unique())
        xs, ys = np.meshgrid(x, y, indexing="ij")
        image = df["Z"].values.reshape(len(x), len(y), order="F")
        return xs, ys, image

    # ======================================================
    # DAT
    # ======================================================

    elif extension == "dat":
        # Auto-detect header by checking if first row can be converted to float
        skiprows = 0
        with open(filename, 'r') as f:
            first_line = f.readline().strip()
            columns = first_line.split()
            second_line = f.readline().strip()
            try:
                # Try to convert first row to floats
                float(first_line.split()[0])
            except (ValueError, IndexError):
                # First row is not numeric, it's a header
                skiprows = 1
            try:
                # Try to convert second row to floats
                float(second_line.split()[0])
            except (ValueError, IndexError):
                # Second row is not numeric, it's a header
                skiprows = 2

        data = np.loadtxt(filename, skiprows=skiprows)
        X = None
        Y = None


        # Determine column indices based on header or default
        if columns[2] == "Intensity":
            X = data[:, 0]
            Y = data[:, 1]
            intensity = data[:, 2]
        
        elif columns is not None and skiprows > 0:
            # Parse header to find H, K, L, and intensity columns
            col_info = parse_column_headers(columns)
            
            if col_info is None:
                raise ValueError(
                    f"Could not parse column headers: {columns}. "
                    "Expected columns like 'H', 'K', 'L', 'Q_H', 'Q_K', 'Q_L', or 'Intensity'"
                )
            
            indices = col_info['indices']
            intensity_ind = col_info['intensity_ind']
            labels = col_info['labels']

            for index in indices:
                h_coeff, k_coeff, l_coeff = index
                if h_coeff != 0:
                    H = h_coeff * data[:, indices.index(index)]
                if k_coeff != 0:
                    K = k_coeff * data[:, indices.index(index)]
                if l_coeff != 0:
                    L = l_coeff * data[:, indices.index(index)]

            intensity = data[:, intensity_ind]

        else:
            print(f"No header found. Assuming column order: H K L Intensity")
            H = data[:, 0]
            K = data[:, 1]
            L = data[:, 2]
            intensity = data[:, 3]

        if X is None or Y is None:
            # Determine which H, K, L correspond to X (in-plane) and Y (out-of-plane)
            if Qin is not None and Qout is not None:
                X, Y = map_HKL_to_XY(H, K, L, Qin, Qout)
            else:
                # Fallback: assume HHL scan
                print("Warning: Qin and Qout not provided. Assuming HHL scan (H→X, L→Y)")
                X = H
                Y = L

        # Cluster coordinates
        x, X = cluster_coordinates(X, coordinate_tolerance)
        y, Y = cluster_coordinates(Y, coordinate_tolerance)

        xs, ys = np.meshgrid(x, y, indexing="ij")

        image = np.full((len(x), len(y)), np.nan)

        x_lookup = {value: i for i, value in enumerate(x)}
        y_lookup = {value: j for j, value in enumerate(y)}

        for xx, yy, zz in zip(X, Y, intensity):
            i = x_lookup[xx]
            j = y_lookup[yy]
            image[i, j] = zz

        return xs, ys, image

    else:
        raise ValueError(f"Unsupported file type: {extension}")


# --------------------------------------------------------
# Save simulation
# --------------------------------------------------------

def save_csv(
    filename,
    xs,
    ys,
    image,
):
    """
    Saves a reciprocal-space map as a CSV file.
    """

    data = np.column_stack(
        (
            xs.ravel(order="F"),
            ys.ravel(order="F"),
            image.ravel(order="F"),
        )
    )

    df = pd.DataFrame(
        data,
        columns=[
            "X",
            "Y",
            "Z",
        ],
    )

    df.to_csv(
        filename,
        index=False,
    )