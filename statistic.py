from PIL import Image, ImageFont, ImageDraw, ImageChops, ImageStat
import numpy as np
import itertools
import os

font_folder = 'data/fonts/google-fonts-cyrillic'
font_list = {}
fonts = []
fonts_len = 0

def load_fonts():
    global font_list, fonts_len

    font_list = {
        font_name: [
            font_ver for font_ver in os.listdir(f'{font_folder}/{font_name}')
        ] for font_name in os.listdir(font_folder)
    }

    for font_name, font_versions in font_list.items():
        #for font_ver in font_versions:
        font_ver = font_versions[0]
        fonts.append(
            ImageFont.truetype(os.path.join(font_folder, font_name, font_ver), size=256)
        )

    fonts_len = len(fonts)

image_c = Image.new('L', (512, 512), 0)
draw_c = ImageDraw.Draw(image_c)
def draw_letter(letter, image_b, font_b):
    draw_c.text((16, 16), letter, 255, font_b)
    
    cropped = image_c.crop(image_c.getbbox())
    image_b.paste(cropped, (
        (image_b.size[0] - cropped.size[0]) // 2,
        (image_b.size[1] - cropped.size[1]) // 2,
    ))

    image_c.paste(0, (0, 0, image_c.size[0], image_c.size[1]))

letter = 'я'
def calc_diff_mat():

    load_fonts()

    image_a = Image.new('L', (512, 512), 0)
    image_b = Image.new('L', (512, 512), 0)
    
    m = np.ndarray((fonts_len, fonts_len), dtype=int)

    for (A, font_a), (B, font_b) in itertools.combinations(enumerate(fonts), 2):
        
        draw_letter(letter, image_a, font_a)
        draw_letter(letter, image_b, font_b)

        pixel_diff = ImageChops.difference(image_a, image_b)
        histogram_diff = round(ImageStat.Stat(pixel_diff).sum[0])
        m[A][B] = histogram_diff
        m[B][A] = histogram_diff
        
        pixel_diff.save(f'ram/{histogram_diff}_{A}_{B}.png')
        #image_a.save(f'ram/{A}.png')

        image_a.paste(0, (0, 0, image_a.size[0], image_a.size[1]))
        image_b.paste(0, (0, 0, image_b.size[0], image_b.size[1]))

    #m = np.array(m)
    #m -= np.min(m)
    #m = m.tolist()

    print('m = ', m.tolist(), file=open('output.py', 'w'))
    return m

from output import m
m = np.array(m)

#m = calc_diff_mat()

""" #TODO: ignore diagonal
print(np.argmin(np.min(m, axis=0)))
print(m.min())

maxrow = m.max(m, axis=1).argmax()
print(maxrow)
print(m[maxrow].argmax())
print(m.max())

print(m[12][12])
"""

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='complete')
model.fit(m)

load_fonts()
samples = []
for i, font in enumerate(fonts):
    image = Image.new('L', (512, 512), 0)
    draw_letter(letter, image, font)
    samples.append(image)
    image.save(f'ram/{i}.png')

n_samples = len(samples)
counts = samples

def file_name(id):
    return f"{id}.{ 'png' if id < n_samples else 'html' }"

for i, merge in enumerate(model.children_):
    blend = Image.blend(counts[merge[0]], counts[merge[1]], 0.5)
    counts.append(blend)
    id = i + n_samples
    blend.save(f'ram/{id}.png')
    with open(f'ram/{id}.html', 'w') as html:
        print(f'<a href="{file_name(merge[0])}"><img src="{merge[0]}.png"></img></a>', file=html)
        print(f'<a href="{file_name(merge[1])}"><img src="{merge[1]}.png"></img></a>', file=html)

"""
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0], dtype=np.int)
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    print(counts.tolist())
    exit()

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
"""