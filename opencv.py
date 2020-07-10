from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from PIL import Image, ImageDraw
import numpy as np
import itertools
from math import sqrt
import os

class ApplePen(BasePen):
    
    def __init__(self, glyphSet):
        super(ApplePen, self).__init__(glyphSet)
        self.points = []
        
    def _moveTo(self, pt):
        self.points.append(pt)

    def _lineTo(self, pt):
        self.points.append(pt)

    def _curveToOne(self, bcp1, bcp2, pt):
        self.points.extend([bcp1, bcp2, pt])

char = 'я'
font_folder = 'data/fonts/google-fonts-cyrillic'
fonts = []

font_list = {
    font_name: [
        font_ver for font_ver in os.listdir(f'{font_folder}/{font_name}')
    ] for font_name in os.listdir(font_folder)
}

for font_name, font_versions in font_list.items():
    font_ver = font_versions[0]
    font = TTFont(f'{font_folder}/{font_name}/{font_ver}')
    font.name = font_name
    fonts.append(font)

def get_points(font):

    glyph_set = font.getGlyphSet()
    glyph = glyph_set[ font.getBestCmap()[ord(char)] ]

    pen = ApplePen(glyph_set)
    glyph.draw(pen)

    points = np.array(pen.points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    bbox = ( x.min(), y.min(), x.max(), y.max() )
    points -= np.array((bbox[0], bbox[1])) # translate to (0, 0)
    points *= np.array((
        250 / (bbox[2] - bbox[0]),
        250 / (bbox[3] - bbox[1])
    ))

    """
    image = Image.new('RGB', (250, 250))
    draw = ImageDraw.Draw(image)
    draw.line(points.flatten().tolist(), fill=255, width=4)
    image.show()
    exit()
    """

    return points

def distance(p1, p2):
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

points = []
for font in fonts:
    #try:
    points.append(get_points(font))
    #except KeyError:
    #    print(font.name)

len_fonts = len(points)
m = np.ndarray((len_fonts, len_fonts))

for (A, points_A), (B, points_B) in itertools.combinations(enumerate(points), 2):

    min_points = points_A if points_A.shape[0] <= points_B.shape[0] else points_B
    max_points = points_A if points_A.shape[0] > points_B.shape[0] else points_B

    """
    image = Image.new('RGB', (250, 250))
    draw = ImageDraw.Draw(image)
    draw.line(points_A.flatten().tolist(), fill=255, width=4)
    draw.line(points_B.flatten().tolist(), fill=125, width=4)
    """

    global_dist = 0
    min_dist = -1
    #min_dist_b = -1
    for point_a in min_points:
        for point_b in max_points:
            dist = distance(point_a, point_b)
            if dist < min_dist or min_dist == -1:
                min_dist = dist
                #min_dist_b = point_b
        """
        draw.line([
            point_a[0].item(), point_a[1].item(),
            min_dist_b[0].item(), min_dist_b[1].item()
        ], fill=(125, 255, 125), width=2)
        """
        global_dist += min_dist
        min_dist = -1
        #min_dist_b = -1

    m[A][B] = m[B][A] = global_dist
    #image.show()

print('m = ', m.tolist(), file=open('output.py', 'w'))
