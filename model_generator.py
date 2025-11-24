import config

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
        
        
def create_model_xml(grid=config.GRID_SIZE, indenter=config.INDENTER_RADIUS):
    """创建模型XML内容"""
    pin_ids = generate_pin_ids(grid)
    mass = (grid[0]-1) * (grid[0]-1) * (grid[2]-1) * config.FLEXCOMP_MASS_MULTIPLIER
    print(f"柔性体质量: {mass:.6f} kg")

    xml_content = f'''
    <mujoco model="Press">
        <include file="scene.xml" />
        <compiler autolimits="true" />
        <option solver="Newton" tolerance="1e-6" timestep="{config.TIMESTEP}" 
            cone="elliptic" integrator="implicitfast"/>
        <size memory="100M" />
        <worldbody>
            <flexcomp name="A" type="grid" count="{grid[0]} {grid[1]} {grid[2]}" 
                spacing="{config.GRID_SPACING} {config.GRID_SPACING} {config.GRID_SPACING}" 
                pos="0 0 {(grid[2]-1)*config.GRID_SPACING/2}"
                radius=".0005" rgba="0 1 1 1" dim="3" mass="{mass}">
                <pin id="{pin_ids}" />
                <contact condim="6" selfcollide="none" solref="0.00001 1" 
                    solimp="0.99999 0.99999 0.001 0.99999 1" gap="0"/>
                <edge damping="1e-1" solimp="0.99999 0.99999 0.001 0.99999 1" solref="0.00001 1"/>
                <elasticity young="2e3" poisson="0.49" />
            </flexcomp>
            <body name="indenter">
                <joint name="soft" type="slide" axis="0 0 1" range="{config.INDENTER_LIMIT_L} {config.INDENTER_LIMIT_H}"
                    damping="{config.JOINT_DAMPING}" stiffness="{config.JOINT_STIFFNESS}"/>
                <geom name="indenter_geom" type="capsule" size="{indenter/2} 0.04"
                    pos="{config.INDENTER_X} {config.INDENTER_Y} {indenter/2+0.04}" density="{config.INDENTER_DENSITY}" />
                <site name="force_sensor_site" type="sphere" size="0.005" pos="0 0 0.005" rgba="1 0 0 1"/>
            </body>
            {generate_sensor_array()}
        </worldbody>
        <actuator>
            <position name="position_control" joint="soft" kp="50000" kv="100" ctrlrange="0.015 0.035"/>
        </actuator>
        <sensor>
            <force name="force_sensor" site="force_sensor_site" />
            <actuatorfrc name="actuator_force" actuator="position_control"/>
            <jointpos name="indenter_position" joint="soft"/>
            <jointvel name="indenter_velocity" joint="soft"/>
        </sensor>

    </mujoco>
    '''
    return xml_content

def save_model_to_file(xml_content, filename=config.MODEL_OUTPUT_FILE):
    """保存模型到文件"""
    with open(filename, 'w') as f:
        f.write(xml_content)
    return filename