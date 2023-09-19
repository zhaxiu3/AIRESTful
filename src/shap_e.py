import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.image_util import load_image
from shap_e.util.notebooks import decode_latent_mesh


class ShapE:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xm = load_model('transmitter', device=self.device)
        self.model = load_model('image300M', device=self.device)
        self.diffusion = diffusion_from_config(load_config('diffusion'))
    
    def predict(self, request_id, image_path, prompt=None):
        batch_size = 1
        guidance_scale = 3.0
        image = load_image(image_path)
        latents = sample_latents(
            batch_size=batch_size,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(images=[image] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        for i, latent in enumerate(latents):
            with open(f'/home/ubuntu/outputs/{request_id}.obj', 'wb') as f:
                decode_latent_mesh(self.xm, latent).tri_mesh().write_obj(f)
