import xml.etree.ElementTree as ET
import os

def strip_visual_collision(input_urdf, output_urdf):
    if not os.path.exists(input_urdf):
        print(f"[ERROR] 输入 URDF 文件不存在：{input_urdf}")
        return

    print(f"[INFO] 正在读取 URDF：{input_urdf}")
    tree = ET.parse(input_urdf)
    root = tree.getroot()

    # 遍历所有 link，删除 visual 和 collision 节点
    for link in root.findall("link"):
        # 删除 <visual> 节点
        for visual in list(link.findall("visual")):
            link.remove(visual)

        # 删除 <collision> 节点
        for collision in list(link.findall("collision")):
            link.remove(collision)

    # 写出新的轻量 URDF
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"[OK] 已生成轻量 IK URDF：{output_urdf}")


if __name__ == "__main__":
    # 修改成你的真实 URDF
    input_urdf = "robot_description/v3/urdf/v3_urdf_251121-2.urdf"
    output_urdf = "robot_description/v3/urdf/v3_urdf_251121-2_ik_new.urdf"

    strip_visual_collision(input_urdf, output_urdf)
