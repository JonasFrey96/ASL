import neptune
import os
import time
print("Start neptune session")
# proxies = {
#   'http': 'http://proxy.ethz.ch:3128',
#   'https': 'http://proxy.ethz.ch:3128',
# }

neptune.init(project_qualified_name='jonasfrey96/ASL', api_token=os.environ["NEPTUNE_API_TOKEN"]) # add your credentials
neptune.create_experiment()
import copy
import matplotlib.pyplot as plt
import numpy as np
def samplemat(dims):
    """Make a matrix with all zeros and increasing elements on the diagonal"""
    aa = np.zeros(dims)
    for i in range(min(dims)):
        aa[i, i] = i
    return aa
fig, ax = plt.subplots()
# Display matrix
ax.matshow(samplemat((15, 15)))

from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image
def get_img_from_fig(fig, dpi=180):
  canvas = FigureCanvasAgg(fig)
  # Retrieve a view on the renderer buffer
  canvas.draw()
  buf = canvas.buffer_rgba()
  # convert to a NumPy array
  buf = np.asarray(buf)
  buf = PIL.Image.fromarray(buf)
  buf = buf.convert('RGB')
  return buf

img = get_img_from_fig(fig)
neptune.log_image('Final', img)



