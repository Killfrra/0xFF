import argparse
import os, sys
from trdg.string_generator import create_strings_from_dict
from trdg.data_generator import FakeTextDataGenerator
import random as rnd


parser = argparse.ArgumentParser(
    description="Generate synthetic text data for text recognition."
)
parser.add_argument('-o', "--output_dir", type=str, nargs="?", help="The output directory", default="output/trdg")
parser.add_argument("-c", "--count", type=int, nargs="?", help="The number of images to be created.", required=True)
parser.add_argument("-f", "--font_dir", type=str, nargs="?", help="Define a font directory to be used")
parser.add_argument("-d", "--dict", type=str, nargs="?", help="Define the dictionary to be used")
parser.add_argument(
    "-i",
    "--image_dir",
    type=str,
    nargs="?",
    help="Define an image directory to use when background is set to image",
    default=os.path.join(os.path.split(os.path.realpath(__file__))[0], "images"),
)

args = parser.parse_args()

lang_dict = []
if os.path.isfile(args.dict):
    with open(args.dict, "r", encoding="utf8", errors="ignore") as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
else:
    sys.exit("Cannot open dict")

font_names = [
    p for p in os.listdir(args.font_dir) if p.endswith(".ttf")
]

strings = create_strings_from_dict(length=2, allow_variable=True, count=len(font_names) * args.count, lang_dict=lang_dict)

for font_name in font_names:
    #out_dir = '%s/%s' % (args.output_dir, font_name[:-4])
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    font = '%s/%s' % (args.font_dir, font_name)

    for i in range(args.count):

        image = FakeTextDataGenerator.generate(
            index = i,
            text = strings[rnd.randrange(0, len(strings))],
            font = font,
            out_dir = None,
            size = 72, #rnd.randint(27, 72),
            extension = '.png', # doesn't matter
            skewing_angle = 10,
            random_skew = True,
            blur = 0,  #TODO: adjust
            random_blur = True,
            background_type = rnd.randint(0, 3),
            distorsion_type = 0, #rnd.randint(0, 3),
            distorsion_orientation = rnd.randint(0, 2),
            is_handwritten = False,
            name_format = 0, # doesn't matter
            width = -1, #TODO: randomize
            alignment = rnd.randint(0, 2),
            text_color = '#282828', #TODO: randomize
            orientation = rnd.randint(0, 1),
            space_width = rnd.uniform(1.0, 2.0),
            character_spacing = rnd.randint(-2, 2), #TODO: ajust
            margins = [0] * 4, #[ rnd.randint(0, 10) for _ in range(4) ],
            fit = rnd.choice([True, False]),
            output_mask = False,
            word_split = rnd.choice([True, False]), #TODO: what is it?
            image_dir = args.image_dir
        )

        image.save('%s/%s_%d.png' % (out_dir, font_name[:-4], i))
        #mask.save('%s/%s_%d_mask.png' % (out_dir, font_name[:-4], i))

        