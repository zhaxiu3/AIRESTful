import torch
from PIL import Image

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.image_util import load_image
from shap_e.util.notebooks import decode_latent_mesh

from utils.sam_utils import sam_init, sam_out_nosave, image_preprocess_nosave, pred_bbox


class ShapE:
    def __init__(self) -> None:
        self.sam = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xm = load_model('transmitter', device=self.device)
        self.model = load_model('image300M', device=self.device)
        self.diffusion = diffusion_from_config(load_config('diffusion'))
    
    def preprocess(self, image_path):
        if self.sam == None:
            self.sam = sam_init()

        image_raw = Image.open(image_path)
        image_raw.thumbnail([512,512], Image.Resampling.LANCZOS)
        image_sam = sam_out_nosave(self.sam, image_raw.convert("RGB"),pred_bbox(image_raw))
        image_256 = image_preprocess_nosave(image_sam)
        torch.cuda.empty_cache()
        return image_256
    
    def predict(self, request_id, image_path, prompt=None, remove_bg=False):
        batch_size = 1
        guidance_scale = 3.0
        
        if remove_bg == True:
            image_256 = self.preprocess(image_path)
            image_256.save(image_path)

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
            with open(f'/home/ubuntu/outputs/{request_id}.ply', 'wb') as f:
                decode_latent_mesh(self.xm, latent).tri_mesh().write_ply(f)
