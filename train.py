import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
)
from renderer import implicit_renderer
from utils.helper_functions import huber, sample_images_at_mc_locs, show_full_render
from model import NeuralRadianceField


def train(device):
    # First move all relevant variables to the correct device.
    renderer_grid = renderer_grid.to(device)
    renderer_mc = renderer_mc.to(device)
    target_cameras = target_cameras.to(device)
    target_images = target_images.to(device)
    target_silhouettes = target_silhouettes.to(device)

    # Set the seed for reproducibility
    torch.manual_seed(1)

    # Instantiate the radiance field model.
    neural_radiance_field = NeuralRadianceField().to(device)

    # Instantiate the Adam optimizer. We set its master learning rate to 1e-3.
    lr = 1e-3
    optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr)

    # We sample 6 random cameras in a minibatch. Each camera
    # emits raysampler_mc.n_pts_per_image rays.
    batch_size = 6

    # 3000 iterations take ~20 min on a Tesla M40 and lead to
    # reasonably sharp results. However, for the best possible
    # results, we recommend setting n_iter=20000.
    n_iter = 3000

    # Init the loss history buffers.
    loss_history_color, loss_history_sil = [], []

    # The main optimization loop.
    for iteration in range(n_iter):      
        # In case we reached the last 75% of iterations,
        # decrease the learning rate of the optimizer 10-fold.
        if iteration == round(n_iter * 0.75):
            print('Decreasing LR 10-fold ...')
            optimizer = torch.optim.Adam(
                neural_radiance_field.parameters(), lr=lr * 0.1
            )
        
        # Zero the optimizer gradient.
        optimizer.zero_grad()
        
        # Sample random batch indices. ## get batch_size here 6 random index from [0, num_view=40]
        batch_idx = torch.randperm(len(target_cameras))[:batch_size]
        
        # Sample the minibatch of cameras.  ## so it returns 6 cameras 
        batch_cameras = FoVPerspectiveCameras(
            R = target_cameras.R[batch_idx], ## 6,3,3
            T = target_cameras.T[batch_idx], 
            znear = target_cameras.znear[batch_idx],
            zfar = target_cameras.zfar[batch_idx],
            aspect_ratio = target_cameras.aspect_ratio[batch_idx],
            fov = target_cameras.fov[batch_idx],
            device = device,
        )
        
        # Evaluate the nerf model.
        rendered_images_silhouettes, sampled_rays = renderer_mc(
            cameras=batch_cameras, 
            volumetric_function=neural_radiance_field
        )
        rendered_images, rendered_silhouettes = (
            rendered_images_silhouettes.split([3, 1], dim=-1)
        )
        
        # Compute the silhouette error as the mean huber
        # loss between the predicted masks and the
        # sampled target silhouettes.
        silhouettes_at_rays = sample_images_at_mc_locs(
            target_silhouettes[batch_idx, ..., None],  ## cos target_silhouettes size =(N,H,W) only
            sampled_rays.xys
        )
        sil_err = huber(
            rendered_silhouettes, 
            silhouettes_at_rays,
        ).abs().mean()

        # Compute the color error as the mean huber
        # loss between the rendered colors and the
        # sampled target images.
        colors_at_rays = sample_images_at_mc_locs(
            target_images[batch_idx], 
            sampled_rays.xys
        )
        color_err = huber(
            rendered_images, 
            colors_at_rays,
        ).abs().mean()
        
        # The optimization loss is a simple
        # sum of the color and silhouette errors.
        loss = color_err + sil_err
        
        # Log the loss history.
        loss_history_color.append(float(color_err))
        loss_history_sil.append(float(sil_err))
        
        # Every 10 iterations, print the current values of the losses.
        if iteration % 10 == 0:
            print(
                f'Iteration {iteration:05d}:'
                + f' loss color = {float(color_err):1.2e}'
                + f' loss silhouette = {float(sil_err):1.2e}'
            )
        
        # Take the optimization step.
        loss.backward()
        optimizer.step()
        
        # Visualize the full renders every 100 iterations.
        if iteration % 100 == 0:
            show_idx = torch.randperm(len(target_cameras))[:1]
            show_full_render(
                neural_radiance_field,
                FoVPerspectiveCameras(
                    R = target_cameras.R[show_idx], 
                    T = target_cameras.T[show_idx], 
                    znear = target_cameras.znear[show_idx],
                    zfar = target_cameras.zfar[show_idx],
                    aspect_ratio = target_cameras.aspect_ratio[show_idx],
                    fov = target_cameras.fov[show_idx],
                    device = device,
                ), 
                target_images[show_idx][0],
                target_silhouettes[show_idx][0],
                loss_history_color,
                loss_history_sil,
            )
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print("device", device)
    else:
        print(
            'Please note that NeRF is a resource-demanding method.'
            + ' Running this notebook on CPU will be extremely slow.'
            + ' We recommend running the example on a GPU'
            + ' with at least 10 GB of memory.'
        )
        device = torch.device("cpu")
    train(device)
if __name__=="__main__":
    main()