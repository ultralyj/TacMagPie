import xml.etree.ElementTree as ET
import random
import argparse

def add_colored_spheres(input_xml, output_xml, num_spheres=20, base_pos=(0, 0, 0.03), 
                       sphere_size=0.01, position_range=0.01, density=1000.0):
    """
    在MuJoCo XML文件中添加彩色小球
    
    参数:
    - input_xml: 输入XML文件路径
    - output_xml: 输出XML文件路径
    - num_spheres: 小球数量
    - base_pos: 基准位置 (x, y, z)
    - sphere_size: 小球半径
    - position_range: 位置随机范围
    - density: 小球密度 (kg/m³)，默认为1000（水的密度）
    """
    
    # 解析XML文件
    tree = ET.parse(input_xml)
    root = tree.getroot()
    
    # 找到worldbody元素
    worldbody = root.find('worldbody')
    if worldbody is None:
        print("错误: 找不到worldbody元素")
        return
    
    # 预定义的颜色列表（RGB格式，0-1范围）
    colors = [
        (1, 0, 0, 1),    # 红色
        (0, 1, 0, 1),    # 绿色
        (0, 0, 1, 1),    # 蓝色
        (1, 1, 0, 1),    # 黄色
        (1, 0, 1, 1),    # 紫色
        (0, 1, 1, 1),    # 青色
        (1, 0.5, 0, 1),  # 橙色
        (0.5, 0, 1, 1),  # 深紫色
        (1, 0.75, 0.8, 1), # 粉色
        (0.5, 0.5, 0.5, 1) # 灰色
    ]
    
    # 创建资产部分（如果不存在）
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    
    # 创建材质
    for i, color in enumerate(colors):
        material_name = f"sphere_material_{i}"
        material = ET.SubElement(asset, 'material')
        material.set('name', material_name)
        material.set('rgba', f"{color[0]} {color[1]} {color[2]} {color[3]}")
    
    # 计算球体的质量和惯性
    # 球体体积: V = (4/3) * π * r³
    # 质量: m = density * V
    # 球体惯性矩: I = (2/5) * m * r²
    r = sphere_size
    mass = density * (4/3) * 3.141592653589793 * r**3
    inertia = (2/5) * mass * r**2
    
    # 添加小球
    for i in range(num_spheres):
        # 生成随机位置偏移
        offset_x = random.uniform(-position_range, position_range)
        offset_y = random.uniform(-position_range, position_range)
        offset_z = random.uniform(0, 0.01)  # 将z方向偏移调整为正值，确保在初始位置之上
        
        pos_x = base_pos[0] + offset_x
        pos_y = base_pos[1] + offset_y
        pos_z = base_pos[2] + offset_z
        
        # 随机选择颜色
        color_index = i % len(colors)
        
        # 创建body元素
        body = ET.SubElement(worldbody, 'body')
        body.set('name', f'sphere_{i+1}')
        body.set('pos', f'{pos_x:.6f} {pos_y:.6f} {pos_z:.6f}')
        
        freejoint = ET.SubElement(body, 'freejoint')
    
        # 创建geom元素
        geom = ET.SubElement(body, 'geom')
        geom.set('type', 'sphere')
        geom.set('size', f'{sphere_size}')
        geom.set('material', f'sphere_material_{color_index}')

    # 添加选项确保重力开启
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    
    # 确保重力设置正确（默认是0 0 -9.81）
    if 'gravity' not in option.attrib:
        option.set('gravity', '0 0 -9.81')
    
    option.set('integrator', 'RK4')
    
    # 美化XML输出
    indent_xml(root)
    
    # 保存修改后的XML
    tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    print(f"成功生成 {num_spheres} 个可自由下落的小球到 {output_xml}")
    print(f"小球质量: {mass:.10f} kg, 半径: {sphere_size} m")
    print(f"重力设置: {option.get('gravity')}")

def indent_xml(elem, level=0, indent="  "):
    """
    美化XML输出，添加缩进
    """
    i = "\n" + level * indent
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            indent_xml(child, level + 1, indent)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


if __name__ == "__main__":
    add_colored_spheres("./2f85.xml", "./ball.xml", num_spheres=20, sphere_size=0.003)