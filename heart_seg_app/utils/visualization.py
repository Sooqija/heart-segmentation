import torch
from torchvision.utils import make_grid

import vtk
import numpy as np

import matplotlib
import matplotlib.colors
import matplotlib.cm as cm

from PIL import Image

def create_custom_cmap(label_values : list, colors : list):
    under_color, over_color = colors[0], colors[-1]
    mid_bounds = [(label_values[i] + label_values[i+1]) / 2 for i in range(len(label_values)-1)]
    custom_cmap = matplotlib.colors.ListedColormap(list(colors)[1:], N=len(colors)).with_extremes(under=under_color, over=over_color)
    norm = matplotlib.colors.BoundaryNorm(mid_bounds, custom_cmap.N - 2)
    return custom_cmap, norm

def vtk_visualize_3d(data : np.ndarray, label_values : list, colors : list, apply_cfilter: bool = False):
    cmap, norm = create_custom_cmap(label_values, colors)
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(data.shape)
    vtk_data.SetSpacing([1, 1, 1])
    vtk_data.SetOrigin([0, 0, 0])
    vtk_data.AllocateScalars(vtk.VTK_INT, 1)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                vtk_data.SetScalarComponentFromFloat(i, j, k, 0, data[i, j, k])

    image = vtk_data

    actors = []
    label_values = label_values[1:]
    renderer = vtk.vtkRenderer()
    for i in range(len(label_values)):

        marching_cubes = vtk.vtkDiscreteMarchingCubes()
        marching_cubes.SetInputData(image)
        marching_cubes.SetValue(0, label_values[i])
        marching_cubes.Update()
        mc_image = marching_cubes.GetOutputPort()

        if apply_cfilter:
            confilter = vtk.vtkPolyDataConnectivityFilter()
            confilter.SetInputConnection(mc_image)
            confilter.SetExtractionModeToLargestRegion()
            confilter.Update()
            mc_image = confilter.GetOutputPort()

        lookup_table = vtk.vtkLookupTable()
        lookup_table.SetNumberOfTableValues(1)
        lookup_table.Build()
        val = vtk.vtkNamedColors().GetColor3d(cmap.colors[i])
        lookup_table.SetTableValue(0, val[0], val[1], val[2], 1.0)

        mapper = vtk.vtkPolyDataMapper()
        mapper.AddInputConnection(mc_image)
        mapper.SetLookupTable(lookup_table)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        actors.append(actor)

        renderer.AddActor(actors[-1])

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.Initialize()
    renderer.Render()
    window.Render()
    interactor.Start()

def vtk_visualize_3d_scan(data : np.ndarray, color_dict : dict, apply_cfilter: bool = False):
    pixel_values = [value[0] for value in color_dict.values()]
    cmap, norm = create_custom_cmap(color_dict)
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(data.shape)
    vtk_data.SetSpacing([1, 1, 1])
    vtk_data.SetOrigin([0, 0, 0])
    vtk_data.AllocateScalars(vtk.VTK_INT, 1)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                vtk_data.SetScalarComponentFromFloat(i, j, k, 0, data[i, j, k])

    image = vtk_data
    
    actors = []
    pixel_values = pixel_values[1:]
    renderer = vtk.vtkRenderer()
    for i in range(len(pixel_values)):

        marching_cubes = vtk.vtkDiscreteMarchingCubes()
        marching_cubes.SetInputData(image)
        marching_cubes.SetValue(0, pixel_values[i])
        marching_cubes.Update()
        mc_image = marching_cubes.GetOutputPort()

        if apply_cfilter:
            confilter = vtk.vtkPolyDataConnectivityFilter()
            confilter.SetInputConnection(mc_image)
            confilter.SetExtractionModeToLargestRegion()
            confilter.Update()
            mc_image = confilter.GetOutputPort()

        lookup_table = vtk.vtkLookupTable()
        lookup_table.SetNumberOfTableValues(1)
        lookup_table.Build()
        val = vtk.vtkNamedColors().GetColor3d(cmap.colors[i])
        lookup_table.SetTableValue(0, val[0], val[1], val[2], 1.0)

        mapper = vtk.vtkPolyDataMapper()
        mapper.AddInputConnection(mc_image)
        mapper.SetLookupTable(lookup_table)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        actors.append(actor)

        renderer.AddActor(actors[-1])

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.Initialize()
    renderer.Render()
    window.Render()
    interactor.Start()
    
def make_grid_image(mode : str, image : torch.Tensor, label : torch.Tensor, prediction : torch.Tensor, label_color_map,idx):
        bone_cmap = cm.get_cmap("bone")
        cmap, norm = create_custom_cmap(label_color_map)

        image = image.squeeze().cpu() # delete batch
        image = image[:,:,idx].T
        image = bone_cmap(image).astype(np.float32) # it converts tensor to numpy rgba image with hwc format
        image = torch.from_numpy(image)
        image = image[:,:,:3] # delete alpha channel
        image = image.permute(2, 0, 1) # hwc -> chw
        
        label = label.squeeze().cpu() # delete batch
        label = torch.argmax(label, dim=0)
        label = label[:,:,idx].T
        label = cmap(norm(label)).astype(np.float32) # it converts tensor to numpy rgba image with hwc format
        label = torch.from_numpy(label)
        label = label[:,:,:3] # delete alpha channel
        label = label.permute(2, 0, 1) # hwc -> chw
        
        prediction = prediction.squeeze().float().cpu() # delete batch
        prediction = torch.softmax(prediction, dim=0).argmax(dim=0)
        prediction = prediction[:,:,idx].T
        prediction = cmap(norm(prediction)).astype(np.float32) # it converts tensor to numpy rgba image with hwc format
        prediction = torch.from_numpy(prediction)
        prediction = prediction[:,:,:3] # delete alpha channel
        prediction = prediction.permute(2, 0, 1) # hwc -> chw
        
        img_grid = make_grid([image, label, prediction])
        if mode == "tensorboard":
            return img_grid
        if mode == "eval":
            img_grid = Image.fromarray((img_grid.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        
        return img_grid