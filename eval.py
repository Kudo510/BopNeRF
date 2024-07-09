import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
# Data structures and functions for rendering
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import FoVPerspectiveCameras
from utils.plot_image_grid import image_grid
from renderer import images_generators, implicit_renderer
from model import NeuralRadianceField

def generate_rotating_nerf(neural_radiance_field, device, n_frames = 50):
    logRs = torch.zeros(n_frames, 3, device=device)
    logRs[:, 1] = torch.linspace(-3.14, 3.14, n_frames, device=device)
    Rs = so3_exp_map(logRs)
    Ts = torch.zeros(n_frames, 3, device=device)
    Ts[:, 2] = 2.7
    frames = []
    print('Rendering rotating NeRF ...')

    target_cameras, target_images = images_generators()
    renderer_grid, renderer_mc = implicit_renderer(target_images)

    for R, T in zip(tqdm(Rs), Ts):
        camera = FoVPerspectiveCameras(
            R=R[None], 
            T=T[None], 
            znear=target_cameras.znear[0],
            zfar=target_cameras.zfar[0],
            aspect_ratio=target_cameras.aspect_ratio[0],
            fov=target_cameras.fov[0],
            device=device,
        )
        # Note that we again render with `NDCMultinomialRaysampler`
        # and the batched_forward function of neural_radiance_field.
        frames.append(
            renderer_grid(
                cameras=camera, 
                volumetric_function=neural_radiance_field.batched_forward,
            )[0][..., :3]
        )
    return torch.cat(frames)

# # First move all relevant variables to the correct device.
# renderer_grid = renderer_grid.to(device)
# renderer_mc = renderer_mc.to(device)
# target_cameras = target_cameras.to(device)
# target_images = target_images.to(device)
# target_silhouettes = target_silhouettes.to(device)

# Set the seed for reproducibility
torch.manual_seed(1)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
# load model ## maybe need to load checkpoints here instead
neural_radiance_field = NeuralRadianceField().to(device)

with torch.no_grad():
    rotating_nerf_frames = generate_rotating_nerf(neural_radiance_field, n_frames=3*5)
    
image_grid(rotating_nerf_frames.clamp(0., 1.).cpu().numpy(), rows=3, cols=5, rgb=True, fill=True)
plt.show()