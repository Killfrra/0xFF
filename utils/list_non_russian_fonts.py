import os
from fontTools.ttLib import TTFont, TTLibError

basepath = 'fonts'
char = ord('я')

for font_name in os.listdir(basepath):
    # https://unix.stackexchange.com/a/258127
    try:
        font = TTFont(os.path.join(basepath, font_name))
        for cmap in font['cmap'].tables:
            if cmap.isUnicode() and char in cmap.cmap:
                break
        else:
            print(font_name)
    except TTLibError as e:
        print(font_name, e)
