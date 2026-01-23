import config
import os

def generate_pin_ids(grid_count=config.GRID_SIZE):
    """生成固定点的ID列表"""
    pin_ids = []
    for i in range(grid_count[0] * grid_count[1]): # 第一层的所有点
        pin_ids.append(str(i * grid_count[2]))
    return " ".join(pin_ids)
def generate_sensor_array(sensor_array = config.SENSOR_ARRAY):
    xml_content = f''
    for i,sens in enumerate(sensor_array):
        xml_content+= f'<site name="sensor_{i}" type="box" size="0.015 0.015 0.005" pos="{sens[0]} {sens[1]} -0.005" rgba="1 1 0 0.3"/>'
    return xml_content
        
        
def create_model_xml():
    """创建模型XML内容"""
    pin_ids = generate_pin_ids(config.GRID_SIZE)
    mass = (config.GRID_SIZE[0]-1) * (config.GRID_SIZE[1]-1) * (config.GRID_SIZE[2]-1) * config.FLEXCOMP_MASS_MULTIPLIER

    xml_content = f'''
    <mujoco model="Press">
        <include file="scene.xml" />
        <compiler autolimits="true" />
        <option solver="Newton" tolerance="1e-8" timestep="{config.TIMESTEP}" 
           integrator="implicitfast"/>
        <size memory="100M" />
        <asset>
            <mesh file="./CNC3018.STL" name="CNC3018" scale="0.001 0.001 0.001" />
            <mesh file="./board.STL" name="board" scale="0.001 0.001 0.001" />
            <mesh file="./rig.STL" name="rig" scale="0.001 0.001 0.001" />
            <mesh file="./indenter_type{config.INDENTER_TYPE}.STL" name="indenter" scale="0.001 0.001 0.001" />
            <mesh file="./press_sensor.STL" name="press_sensor" scale="0.001 0.001 0.001" />
            <mesh file="./press_outer.STL" name="press_outer" scale="0.001 0.001 0.001" />
        </asset>
        <worldbody>
            <body name="cnc3018" pos="-0.1948 -0.1628 0" euler="0 0 0">
                <geom type="mesh" mesh="CNC3018" name="cnc3018_geom" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>
                <body name="board" pos="0.04 0.27 0.06" quat="0.717 0.717 0 0">
                    <geom type="mesh" mesh="board" name="board_geom" rgba="0.4 0.4 0.4 1" contype="0" conaffinity="0"/>
                    <body name="rig" pos="0.12 0.027 0.074" quat="0.717 0.717 0 0">
                        <geom type="mesh" mesh="rig" name="rig_geom" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                            <body name="cube" pos="0.02 -0.01 0.01">
                                <geom name="cube_geom" type="box" size="0.01 0.01 0.01" rgba="0.8 0.2 0.2 1" contype="0" conaffinity="0"/>
                            </body>
                    </body>
                </body>
            </body>
            <flexcomp name="A" type="grid" count="{config.GRID_SIZE[0]} {config.GRID_SIZE[1]} {config.GRID_SIZE[2]}" 
                spacing="{config.GRID_SPACING} {config.GRID_SPACING} {config.GRID_SPACING}" 
                pos="0 0 {0.09+ (config.GRID_SIZE[2]-1)*config.GRID_SPACING/4}"
                radius=".0005" rgba="0 1 1 0.7" dim="3" mass="{mass}">
                <pin id="{pin_ids}" />
                 <contact condim="6" selfcollide="none" solref="0.001 0.9"
                    solimp="0.99999 0.99999 0.001 0.99999 1"/>
                <edge damping="1e-1" />
                <elasticity young="1.2e5" poisson="0.49" damping="1e-6"/>
            </flexcomp>
            <body name="indenter" pos="{config.INDENTER_X} {config.INDENTER_Y} 0.118" quat="0 1 0 0">
            <joint name="indenter_slider" type="slide" axis="0 0 1" range="0 0.013"
                damping="1" stiffness="0" />
            <geom type="mesh" mesh="indenter"  pos="-0.0173 0.0173 0" quat="0.717 0.717 0 0" name="indenter_geom" rgba="0 1 0 0.3"  solref="0.001 0.9"/>
            <site name="force_sensor_site" type="sphere" size="0.0005" pos="0 0 0.005"
                rgba="1 0 0 1" />
            <body name="outer" pos="0 0 0" quat="0 0 0 1">
                    <geom type="mesh" mesh="press_outer" name="press_outer_geom" pos="-0.022 0.022 -0.05" quat="0.717 0.717 0 0" rgba="1 0 0 0.2" />
            </body>
        </body>
        </worldbody>
        <actuator>
            <position name="position_control" joint="indenter_slider" kp="10000000" kv="10000" ctrlrange="0.0 0.013"/>
        </actuator>
        <sensor>
            <force name="force_sensor" site="force_sensor_site" />
            <actuatorfrc name="actuator_force" actuator="position_control"/>
            <jointpos name="indenter_position" joint="indenter_slider"/>
            <jointvel name="indenter_velocity" joint="indenter_slider"/>
        </sensor>

    </mujoco>
    '''
    return xml_content

def save_model_to_file(xml_content, filename=None):
    """保存模型到文件，支持自定义文件名"""
    if filename is None:
        filename = config.MODEL_OUTPUT_FILE
    else:
        filename = os.path.join("temp_models", filename)
        os.makedirs("temp_models", exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write(xml_content)
    return filename