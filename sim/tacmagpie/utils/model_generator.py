import os
import config


def generate_pin_ids(grid_count=config.GRID_SIZE):
    """
    Generate a space-separated list of pinned node IDs for the flex grid.

    Parameters
    ----------
    grid_count : tuple
        Grid resolution in (x, y, z).

    Returns
    -------
    str
        Space-separated pin node indices.
    """
    pin_ids = []
    for i in range(grid_count[0] * grid_count[1]):
        pin_ids.append(str(i * grid_count[2]))
    return " ".join(pin_ids)


def generate_sensor_sites(sensor_array=config.SENSOR_ARRAY):
    """
    Generate XML snippets for sensor sites.

    Parameters
    ----------
    sensor_array : list
        List of sensor positions [[x, y], ...].

    Returns
    -------
    str
        XML string defining MuJoCo <site> elements.
    """
    xml_content = ""
    for i, sens in enumerate(sensor_array):
        xml_content += (
            f'<site name="sensor_{i}" type="box" '
            f'size="0.015 0.015 0.005" '
            f'pos="{sens[0]} {sens[1]} -0.005" '
            f'rgba="1 1 0 0.3"/>\n'
        )
    return xml_content


def load_xml_template(template_path):
    """
    Load an XML template file from disk.

    Parameters
    ----------
    template_path : str
        Path to the XML template file.

    Returns
    -------
    str
        Raw XML template content.
    """
    with open(template_path, "r") as f:
        return f.read()


def create_model_xml(grid=config.GRID_SIZE, template_path=None):
    """
    Create the final MuJoCo model XML by filling a template.

    Parameters
    ----------
    grid : tuple
        Grid resolution in (x, y, z).
    template_path : str or None
        Path to the XML template file. If None, use default.

    Returns
    -------
    str
        Fully populated MuJoCo XML string.
    """
    if template_path is None:
        template_path = config.MODEL_TEMPLATE_FILE

    pin_ids = generate_pin_ids(grid)
    sensor_sites = generate_sensor_sites()

    flex_mass = (
        (grid[0] - 1)
        * (grid[1] - 1)
        * (grid[2] - 1)
        * config.FLEXCOMP_MASS_MULTIPLIER
    )

    print(f"Flex body mass: {flex_mass:.6f} kg")

    template = load_xml_template(template_path)

    xml_content = template.format(
        GRID_X=grid[0],
        GRID_Y=grid[1],
        GRID_Z=grid[2],
        GRID_SPACING=config.GRID_SPACING,
        FLEX_MASS=flex_mass,
        PIN_IDS=pin_ids,
        INDENTER_X=config.INDENTER_X,
        INDENTER_Y=config.INDENTER_Y,
        TIMESTEP=config.TIMESTEP,
        SENSOR_SITES=sensor_sites,
    )

    return xml_content


def save_model_to_file(xml_content, filename=None):
    """
    Save the generated MuJoCo XML to disk.

    Parameters
    ----------
    xml_content : str
        Full XML content of the MuJoCo model.
    filename : str or None
        Output filename. Uses default from config if None.

    Returns
    -------
    str
        Path to the saved XML file.
    """
    if filename is None:
        filename = config.MODEL_OUTPUT_FILE
    else:
        os.makedirs("temp_models", exist_ok=True)
        filename = os.path.join("temp_models", filename)

    with open(filename, "w") as f:
        f.write(xml_content)

    return filename
