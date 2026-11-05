"""
Parsing das anotações XML do LSID-Beans.
Usado por todos os datasets (com ou sem segmentação).
"""
import xml.etree.ElementTree as ET


def strip_known_suffix(name: str) -> str:
    suffixes = ("_area", "_segmentation", "_annotation", "_image")
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break
    return name


def parse_annotation(xml_path: str) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_name = root.findtext("image-name")
    obs = root.findtext("observation", default="")

    size_node = root.find("image-size")
    height = int(size_node.findtext("height"))
    width = int(size_node.findtext("width"))

    objects = root.find("objects")
    leaf_number = objects.findtext("leaf-number")

    marker = objects.find("marker")
    marker_real_area = float(marker.findtext("marker-real-area"))
    marker_projected_area = float(marker.findtext("marker-projected-area"))

    leaf = objects.find("leaf")
    dims = leaf.find("dimensions")

    leaf_area = float(dims.findtext("leaf-area"))
    leaf_width = float(dims.findtext("leaf-width"))
    leaf_length = float(dims.findtext("leaf-length"))
    leaf_perimeter = float(dims.findtext("leaf-perimeter"))

    return {
        "image_name": image_name,
        "observation": obs,
        "height": height,
        "width": width,
        "leaf_number": leaf_number,
        "marker_real_area": marker_real_area,
        "marker_projected_area": marker_projected_area,
        "leaf_area": leaf_area,
        "leaf_width": leaf_width,
        "leaf_length": leaf_length,
        "leaf_perimeter": leaf_perimeter,
    }
