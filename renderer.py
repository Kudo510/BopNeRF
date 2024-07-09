from pytorch3d.renderer import (
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer
)



from utils.plot_image_grid import image_grid
from utils.generate_cow_renders import generate_cow_renders

def images_generators():
    """ Generate images of the scene and masks¶
    The following cell generates our training data. It renders the cow mesh from the fit_textured_mesh.ipynb tutorial from several viewpoints and returns:
        A batch of image and silhouette tensors that are produced by the cow mesh renderer.
        A set of cameras corresponding to each render. """

    target_cameras, target_images, target_silhouettes = generate_cow_renders(num_views=40, azimuth_range=180)
    print(f'Generated {len(target_images)} images/silhouettes/cameras.')
    return target_cameras, target_images
    

def implicit_renderer(target_images):
    """ The renderer is composed of a raymarcher and a raysampler.

    The raysampler is responsible for emitting rays from image pixels and sampling the points along them. Here, we use two different raysamplers:
    MonteCarloRaysampler is used to generate rays from a random subset of pixels of the image plane. The random subsampling of pixels is carried out during training to decrease the memory consumption of the implicit model.
    NDCMultinomialRaysampler which follows the standard PyTorch3D coordinate grid convention (+X from right to left; +Y from bottom to top; +Z away from the user). In combination with the implicit model of the scene, NDCMultinomialRaysampler consumes a large amount of memory and, hence, is only used for visualizing the results of the training at test time.
    The raymarcher takes the densities and colors sampled along each ray and renders each ray into a color and an opacity value of the ray's source pixel. Here we use the EmissionAbsorptionRaymarcher which implements the standard Emission-Absorption raymarching algorithm. """
    # render_size describes the size of both sides of the 
    # rendered images in pixels. Since an advantage of 
    # Neural Radiance Fields are high quality renders
    # with a significant amount of details, we render
    # the implicit function at double the size of 
    # target images.
    render_size = target_images.shape[1] * 2

    # Our rendered scene is centered around (0,0,0) 
    # and is enclosed inside a bounding box
    # whose side is roughly equal to 3.0 (world units).
    volume_extent_world = 3.0

    # 1) Instantiate the raysamplers.

    # Here, NDCMultinomialRaysampler generates a rectangular image
    # grid of rays whose coordinates follow the PyTorch3D
    # coordinate conventions.
    raysampler_grid = NDCMultinomialRaysampler(
        image_height=render_size,
        image_width=render_size,
        n_pts_per_ray=128,
        min_depth=0.1,
        max_depth=volume_extent_world,
    )

    # MonteCarloRaysampler generates a random subset 
    # of `n_rays_per_image` rays emitted from the image plane.
    raysampler_mc = MonteCarloRaysampler(
        min_x = -1.0,
        max_x = 1.0,
        min_y = -1.0,
        max_y = 1.0,
        n_rays_per_image=750,
        n_pts_per_ray=128,
        min_depth=0.1,
        max_depth=volume_extent_world,
    )

    # 2) Instantiate the raymarcher.
    # Here, we use the standard EmissionAbsorptionRaymarcher 
    # which marches along each ray in order to render
    # the ray into a single 3D color vector 
    # and an opacity scalar.
    raymarcher = EmissionAbsorptionRaymarcher()

    # Finally, instantiate the implicit renders
    # for both raysamplers.
    renderer_grid = ImplicitRenderer(
        raysampler=raysampler_grid, raymarcher=raymarcher,
    )
    renderer_mc = ImplicitRenderer(
        raysampler=raysampler_mc, raymarcher=raymarcher,
    )

    return renderer_grid, renderer_mc



