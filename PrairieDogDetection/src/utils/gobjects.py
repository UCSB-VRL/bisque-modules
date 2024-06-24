from bqapi import BQFactory, BQSession, BQGObject
from lxml import etree
import matplotlib

import time

from typing import Optional


def add_rectangle_to_gobj(gobj: etree, box: list, id_color: Optional[list] = None) -> None:
    """Adds a rectangle gobject to a parent gobject

    Args:
        gobj (etree): etree gobject
        box (list): bounding box coordinates in the form [x1, y1, x2, y2]
        id_color (list, optional):  array of RGB values between 0 and 1 Defaults to None.
    """

    rectangle = etree.SubElement(gobj, "rectangle")
    
    # add vertices to rectangle
    # upper left
    vertex = etree.SubElement(rectangle, "vertex")
    vertex.set("index", str(0))
    vertex.set("t", str(0.0))
    vertex.set("x", str(box[2]))
    vertex.set("y", str(box[3]))

    vertex.set("z", str(0.0))

    # Bottom right
    vertex = etree.SubElement(rectangle, "vertex")
    vertex.set("index", str(1))
    vertex.set("t", str(0.0))
    vertex.set("x", str(box[4]))
    vertex.set("y", str(box[5]))
    vertex.set("z", str(0.0))

    if id_color:
        t = etree.SubElement(
            rectangle,
            "tag",
            name="color",
            type="color",
            value=matplotlib.colors.to_hex(id_color),
        )
    return


def create_gobject(boxes: dict, color_map: dict, classes: dict, name: Optional[str] = None) -> etree:
    """ Creates a gobject etree from a dictionary of classes 

    Args:
        boxes (dict of int: list): Maps id to list of boxes for that id ex: {0: [id, x1, y1, x2, y2]}
        color_map (dict of int: array): Maps id to colormap ex: {0: [0.1, 0.1, 0.1]} 
        name (Optional[str], optional): _description_. Defaults to None.

    Returns:
        etree: returns a gobject etree with format:
        <gobject name="name">
            <gobject>
                <rectangle>
                    <vertex index="0" t="0.0" x="x1" y="y1" z="0.0"/>
                    <vertex index="1" t="0.0" x="x2" y="y2" z="0.0"/>
                </rectangle>
            </gobject>
        </gobject>

    """

    # make the name of the outter gobject the current time
    if not name:
        ts = time.gmtime()
        name = time.strftime("%Y-%m-%d %H:%M:%S", ts)

    # create initial parent gobject
    gobj = etree.Element("gobject", name=name)
    etree.SubElement(gobj, "tag", name="color", value=matplotlib.colors.to_hex([0, 0, 0]))

    # loop through each id
    for id in boxes.keys():
        # get the color for the id
        color = color_map[id]

        # create a gobject for each id
        id_gobj = etree.SubElement(gobj, "gobject", name=classes[id])
        etree.SubElement(id_gobj, "tag", name="color", value=matplotlib.colors.to_hex(color))

        # loop through each box
        for box in boxes[id]:
            # create a gobject for each box
            add_rectangle_to_gobj(id_gobj, box)

    return gobj
            

def add_gobject_to_image(bq: BQSession, uri: str, gobj: etree) -> None:
    """
    Uses the BQAPI to add a gobject to an image already existing in the database.

    Args:
        bq (BQSession): bisque user session
        uri (str): URI to bisque image ex: 00-xxxxxxxxxxxx
        gobj (etree): gobject xml to be added to the image
    """

    # load bisque image
    image = bq.load(uri)

    # add gobject to image
    image.add_gob(gob=gobj)

    # save image
    bq.save(image)

    return

def create_gobject_and_add_to_image(bq: BQSession, uri: str, boxes: dict, color_map: dict, classes: dict, name: Optional[str] = None) -> None:
    """ Creates a gobject and adds it to an image

    Args:
        bq (BQSession):  bisque user session
        uri (str): URI to bisque image ex: 00-xxxxxxxxxxxx
        classes (dict of int: array): Maps id to list of boxes for that id ex: {0: [id, x1, y1, x2, y2]}
        color_map (dict): Maps id to colormap ex: {0: [0.1, 0.1, 0.1]}
        name (Optional[str], optional): the name of the gobject. Defaults to the current timestamp
    """

    # create gobject
    gobj = create_gobject(boxes, color_map, classes, name)
    gob = BQFactory(bq).from_etree(gobj)

    # add gobject to image
    add_gobject_to_image(bq, uri, gob)

    return

