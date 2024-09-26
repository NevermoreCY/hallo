# pylint: disable=E1101,C0415,W0718,R0801
# scripts/train_stage1.py
"""
This is the main training script for stage 1 of the project. 
It imports necessary packages, defines necessary classes and functions, and trains the model using the provided configuration.

The script includes the following classes and functions:

1. Net: A PyTorch model that takes noisy latents, timesteps, reference image latents, face embeddings, 
   and face masks as input and returns the denoised latents.
3. log_validation: A function that logs the validation information using the given VAE, image encoder, 
   network, scheduler, accelerator, width, height, and configuration.
4. train_stage1_process: A function that processes the training stage 1 using the given configuration.

The script also includes the necessary imports and a brief description of the purpose of the file.
"""

import argparse
import copy
import logging
import math
import os
import random
import warnings
from datetime import datetime

import cv2
import diffusers
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from insightface.app import FaceAnalysis
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from tqdm.auto import tqdm

from hallo.animate.face_animate_static import StaticPipeline
from hallo.datasets.mask_image import FaceMaskDataset
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.mutual_self_attention import ReferenceAttentionControl
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.util import (compute_snr, delete_additional_ckpt,
                              import_filename, init_output_dir,
                              load_checkpoint, move_final_checkpoint,
                              save_checkpoint, seed_everything)

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    """
    The Net class defines a neural network model that combines a reference UNet2DConditionModel, 
    a denoising UNet3DConditionModel, a face locator, and other components to animate a face in a static image.

    Args:
        reference_unet (UNet2DConditionModel): The reference UNet2DConditionModel used for face animation.
        denoising_unet (UNet3DConditionModel): The denoising UNet3DConditionModel used for face animation.
        face_locator (FaceLocator): The face locator model used for face animation.
        reference_control_writer: The reference control writer component.
        reference_control_reader: The reference control reader component.
        imageproj: The image projection model.

    Forward method:
        noisy_latents (torch.Tensor): The noisy latents tensor.
        timesteps (torch.Tensor): The timesteps tensor.
        ref_image_latents (torch.Tensor): The reference image latents tensor.
        face_emb (torch.Tensor): The face embeddings tensor.
        face_mask (torch.Tensor): The face mask tensor.
        uncond_fwd (bool): A flag indicating whether to perform unconditional forward pass.

    Returns:
        torch.Tensor: The output tensor of the neural network model.
    """

    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        reference_control_writer: ReferenceAttentionControl,
        reference_control_reader: ReferenceAttentionControl,
        imageproj: ImageProjModel,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.imageproj = imageproj

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        face_emb,
        face_mask,
        uncond_fwd: bool = False,
    ):
        """
        Forward pass of the model.
        Args:
            self (Net): The model instance.
            noisy_latents (torch.Tensor): Noisy latents.
            timesteps (torch.Tensor): Timesteps.
            ref_image_latents (torch.Tensor): Reference image latents.
            face_emb (torch.Tensor): Face embedding.
            face_mask (torch.Tensor): Face mask.
            uncond_fwd (bool, optional): Unconditional forward pass. Defaults to False.

        Returns:
            torch.Tensor: Model prediction.
        """

        face_emb = self.imageproj(face_emb)
        face_mask = face_mask.to(device="cuda")
        face_mask_feature = self.face_locator(face_mask)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=face_emb,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            mask_cond_fea=face_mask_feature,
            encoder_hidden_states=face_emb,
        ).sample

        return model_pred


def get_noise_scheduler(cfg: argparse.Namespace):
    """
    Create noise scheduler for training

    Args:
        cfg (omegaconf.dictconfig.DictConfig): Configuration object.

    Returns:
        train noise scheduler and val noise scheduler
    """
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    return train_noise_scheduler, val_noise_scheduler


def log_validation(
    vae,
    net,
    scheduler,
    width,
    height,
    imageproj,
    cfg,
    save_dir,
    global_step,
    face_analysis_model_path,
):
    """
    Log validation generation image.

    Args:
        vae (nn.Module): Variational Autoencoder model.
        net (Net): Main model.
        scheduler (diffusers.SchedulerMixin): Noise scheduler.
        width (int): Width of the input images.
        height (int): Height of the input images.
        imageproj (nn.Module): Image projection model.
        cfg (omegaconf.dictconfig.DictConfig): Configuration object.
        save_dir (str): directory path to save log result.
        global_step (int): Global step number.

    Returns:
        None
    """
    # logger.info("Running validation... ")
    print("flag - 0")

    reference_unet = net.reference_unet
    denoising_unet = net.denoising_unet
    face_locator = net.face_locator
    print("flag - 0 - 1")

    generator = torch.manual_seed(42)

    print("flag - 0 - 2")
    image_enc = FaceAnalysis(
        name="",
        root=face_analysis_model_path,
        # providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        providers=["CPUExecutionProvider"],
    )
    print("flag - 1")
    image_enc.prepare(ctx_id=0, det_size=(640, 640))
    print("flag - 2")

    pipe = StaticPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        face_locator=face_locator,
        scheduler=scheduler,
        imageproj=imageproj,
    )
    print("flag - 3")


    pil_images = []
    for ref_image_path, mask_image_path in zip(cfg.ref_image_paths, cfg.mask_image_paths):
        print(f"{ref_image_path=}")
        print(f"{mask_image_path=}")
        # for mask_image_path in mask_image_paths:
        mask_name = os.path.splitext(
            os.path.basename(mask_image_path))[0]
        ref_name = os.path.splitext(
            os.path.basename(ref_image_path))[0]
        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        mask_image_pil = Image.open(mask_image_path).convert("RGB")

        # Prepare face embeds
        face_info = image_enc.get(
            cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (
            x['bbox'][3] - x['bbox'][1]))[-1]  # only use the maximum face
        face_emb = torch.tensor(face_info['embedding'])
        face_emb = face_emb.to(
            imageproj.device, imageproj.dtype)

        image = pipe(
            ref_image_pil,
            mask_image_pil,
            width,
            height,
            20,
            3.5,
            face_emb,
            generator=generator,
        ).images
        image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
        res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
        # Save ref_image, src_image and the generated_image
        w, h = res_image_pil.size
        canvas = Image.new("RGB", (w * 3, h), "white")
        ref_image_pil = ref_image_pil.resize((w, h))
        mask_image_pil = mask_image_pil.resize((w, h))
        canvas.paste(ref_image_pil, (0, 0))
        canvas.paste(mask_image_pil, (w, 0))
        canvas.paste(res_image_pil, (w * 2, 0))

        out_file = os.path.join(
            save_dir, f"{global_step:06d}-{ref_name}_{mask_name}.jpg"
        )
        canvas.save(out_file)

    del pipe
    del net
    torch.cuda.empty_cache()

    return pil_images



def load_config(config_path: str) -> dict:
    """
    Loads the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """

    if config_path.endswith(".yaml"):
        return OmegaConf.load(config_path)
    if config_path.endswith(".py"):
        return import_filename(config_path).cfg
    raise ValueError("Unsupported format for config file")

def test_stage1(cfg):

    # TODO: add data path

    weight_dtype = torch.float32
    print("0-0-0")

    # create model
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
        use_landmark=False
    ).to(device="cuda", dtype=weight_dtype)
    imageproj = ImageProjModel(
        cross_attention_dim=denoising_unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    ).to(device="cuda", dtype=weight_dtype)
    print("0-0-1")

    if cfg.face_locator_pretrained:
        face_locator = FaceLocator(
            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
        ).to(device="cuda", dtype=weight_dtype)
        miss, _ = face_locator.load_state_dict(
            cfg.face_state_dict_path, strict=False)
        logger.info(f"Missing key for face locator: {len(miss)}")
    else:
        face_locator = FaceLocator(
            conditioning_embedding_channels=320,
        ).to(device="cuda", dtype=weight_dtype)

    # load module weight from stage 1
    stage1_ckpt_dir = cfg.stage1_ckpt_dir
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, "denoising_unet.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, "reference_unet.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    face_locator.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, "face_locator.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    imageproj.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, "imageproj.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    # Freeze
    print("0-0-2")

    vae.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    reference_unet.requires_grad_(False)
    imageproj.requires_grad_(False)
    face_locator.requires_grad_(False)
    print("0-0-3")

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )
    print("0-0-4")

    net = Net(
        reference_unet,
        denoising_unet,
        face_locator,
        reference_control_writer,
        reference_control_reader,
        imageproj,
    ).to(dtype=weight_dtype)

    # get noise scheduler
    train_noise_scheduler, val_noise_scheduler = get_noise_scheduler(cfg)

    print("0-0-5")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(cfg.seed)
    print("0-0-6")

    log_validation(
        vae=vae,
        net=net,
        scheduler=val_noise_scheduler,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        imageproj=imageproj,
        cfg=cfg,
        save_dir="examples/results_stage1",
        global_step=30000,
        face_analysis_model_path=cfg.face_analysis_model_path
    )


if __name__ == "__main__":
    # file tree
    # examples
    # - reference_images
    # - results_stage1
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="./configs/train/stage1.yaml")
    args = parser.parse_args()

    # testing imgs: change configs/train/stage1.yaml:ref_image_paths

    # model ckpt file 
    stage1_ckpt_dir = "exp_output/stage1"

    try:
        config = load_config(args.config)
        config.stage1_ckpt_dir = stage1_ckpt_dir
        test_stage1(config)
    except Exception as e:
        logging.error("Failed to execute the training process: %s", e)
