import ipyvolume.pylab as p3
import ipyvolume
import ipywidgets as widgets
import SimpleITk as sitk
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()

def show3D(filePath):
    image = sitk.ReadImage(filePath)
    img = sitk.GetArrayFromImage(image)
    
    display(ipyvolume.volume.quickvolshow(img))

def showSlice(filePath):
    image = sitk.ReadImage(filePath)
    img = sitk.GetArrayFromImage(image)
    zSlice = img[1, :, :]
    pl = figure(
        plot_width=400, 
        plot_height=400, 
        x_range=(0, 10), 
        y_range=(0, 10))
    slicer = pl.image(
        image = [zSlice], 
        x=[0], 
        y=[0], 
        dw=[10], 
        dh=[10])
    
    def update(z):
        zSlicer = img[z, :, :]
        slicer.data_source.data['image'] = [zSlicer]
        push_notebook()
        
    show(pl, notebook_handle = True)
    play = widgets.Play(
        value=50,
        min=0,
        max=154,
        step=1,
        description="Press play",
        disabled=False)
    slider = widgets.IntSlider()
    widgets.jslink((play, 'value'), (slider, 'value'))
    interact(update, z = play)
    display(slider)