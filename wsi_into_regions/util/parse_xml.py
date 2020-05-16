### Parse XML file containing tumor annotations into a list
### @author: Steve Yang

# System imports
import xml.etree.ElementTree as et


def parse_xml(xml_path):
    '''
    Function to parse coordinates in XML format into a list of tuples
    Arguments:
       - xml_filename: XML filename
       - xml_directory: directory where the XML is located
    
    Returns:
       - coordinates: a list of tuples containing coordinates, divided
                      into segments ([[segment_1], [segment_2], etc])
                      where each segment consists of (x, y) tuples
    '''
    with open(xml_path, 'rt') as f:
        tree = et.parse(f)
        root = tree.getroot()

    # Get the number of coordinates lists inside the xml file
    count = 0
    for Annotation in root.findall('.//Annotation'):
        count += 1

    # Make a list of tuples containing the coordinates
    temp = []
    for Coordinate in root.findall('.//Coordinate'):
        order = float(Coordinate.get('Order'))
        x = float(Coordinate.get('X'))
        y = float(Coordinate.get('Y'))
        temp.append((order, x,y))

    # Separate list of tuples into lists depending on how many segments are annotated
    coordinates = [[] for i in range(count)]
    i = -1
    for j in range(len(temp)):
        if temp[j][0] == 0:
            i += 1
        x = temp[j][1]
        y = temp[j][2]
        coordinates[i].append((x,y))
    
    return coordinates

    
        
    