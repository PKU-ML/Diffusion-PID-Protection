import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers import AutoencoderKL


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of updating steps",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=12.75,
        help='pertubation budget'
    )
    parser.add_argument(
        '--step_size',
        type=float,
        default=1/255,
        help='step size of each update'
    )
    parser.add_argument(
        '--attack_type',
        choices=['var', 'mean', 'KL', 'add-log', 'latent_vector', 'add'],
        help='what is the attack target'
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    return args
      

class PIDDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        size=512,
        center_crop=False
    ):
        self.size = size
        self.center_crop = center_crop
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.image_transforms = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),])

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example['index'] = index % self.num_instance_images
        example['pixel_values'] = self.image_transforms(instance_image)
        return example


def main(args):
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
    weight_dtype = torch.float32
    device = torch.device('cuda')
    
    # VAE encoder
    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    # Dataset and DataLoaders creation:
    dataset = PIDDataset(
        instance_data_root=args.instance_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, # some parts of code don't support batching
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    # Wrapper of the perturbations generator
    class AttackModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            to_tensor = transforms.ToTensor()
            self.epsilon = args.eps/255
            self.delta = [torch.empty_like(to_tensor(Image.open(path))).uniform_(-self.epsilon, self.epsilon) 
                          for path in dataset.instance_images_path]
            self.size = dataset.size
        
        def forward(self, vae, x, index, poison=False):
            # Check whether we need to add perturbation
            if poison:
                self.delta[index].requires_grad_(True)
                x = x + self.delta[index].to(dtype=weight_dtype)
            
            # Normalize to [-1, 1]
            input_x = 2 * x - 1
            return vae.encode(input_x.to(device))
        
    attackmodel = AttackModel()
    
    # Just to zero-out the gradient
    optimizer = torch.optim.SGD(attackmodel.delta, lr=0)
    
    # Progress bar
    progress_bar = tqdm(range(0, args.max_train_steps), desc="Steps")

    # Make sure the dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start optimizing the perturbation
    for step in progress_bar:
        
        total_loss = 0.0
        for batch in dataloader:
            # Save images 
            if step%25 == 0:    
                to_image = transforms.ToPILImage()
                for i in range(0, len(dataset.instance_images_path)):
                    img = dataset[i]['pixel_values']
                    img = to_image(img + attackmodel.delta[i])
                    img.save(os.path.join(args.output_dir, f"{i}.png"))


            # Select target loss
            clean_embedding = attackmodel(vae, batch['pixel_values'], batch['index'], False)
            poison_embedding = attackmodel(vae, batch['pixel_values'], batch['index'], True)
            clean_latent = clean_embedding.latent_dist
            poison_latent = poison_embedding.latent_dist
            
            if args.attack_type == 'var':
                loss = F.mse_loss(clean_latent.std, poison_latent.std, reduction="mean") 
            elif args.attack_type == 'mean':    
                loss = F.mse_loss(clean_latent.mean, poison_latent.mean, reduction="mean") 
            elif args.attack_type == 'KL':
                sigma_2, mu_2 = poison_latent.std, poison_latent.mean
                sigma_1, mu_1 = clean_latent.std, clean_latent.mean
                KL_diver = torch.log(sigma_2 / sigma_1) - 0.5 + (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (2 * sigma_2 ** 2)
                loss = KL_diver.flatten().mean()
            elif args.attack_type == 'latent_vector':
                clean_vector = clean_latent.sample()
                poison_vector = poison_latent.sample()
                loss = F.mse_loss(clean_vector, poison_vector, reduction="mean") 
            elif args.attack_type == 'add':
                loss_2 = F.mse_loss(clean_latent.std, poison_latent.std, reduction="mean") 
                loss_1 = F.mse_loss(clean_latent.mean, poison_latent.mean, reduction="mean") 
                loss = loss_1 + loss_2
            elif args.attack_type == 'add-log':
                loss_1 = F.mse_loss(clean_latent.var.log(), poison_latent.var.log(), reduction="mean")
                loss_2 = F.mse_loss(clean_latent.mean, poison_latent.mean, reduction='mean')
                loss = loss_1 + loss_2
                
                    
            optimizer.zero_grad()
            loss.backward()
            
            # Perform PGD update on the loss
            delta = attackmodel.delta[batch['index']]
            delta.requires_grad_(False)
            delta += delta.grad.sign() * 1/255
            delta = torch.clamp(delta, -attackmodel.epsilon, attackmodel.epsilon)
            delta = torch.clamp(delta, -batch['pixel_values'].detach().cpu(), 1-batch['pixel_values'].detach().cpu())
            attackmodel.delta[batch['index']] = delta.detach().squeeze(0)

            total_loss += loss.detach().cpu()

        # Logging steps
        logs = {"loss": total_loss.item()}
        progress_bar.set_postfix(**logs)
            

if __name__ == "__main__":
    args = parse_args()
    main(args)
