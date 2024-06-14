import torch, os, clip
from argparse import ArgumentParser
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from piq import ssim, psnr
from torchvision import transforms
import torch_fidelity as fid

##########################################################
# Here we implement the whole evaluation pipeline
# Load images from a folder and compute six metrics
# 1. FDS:      Face Detection similarity
# 2. PSNR:     Peak Signal-to-Noise Ratio
# 3. SSIM:     Structural Similarity Index Measure
# 4. CLIP-IQS: Image Quality Evaluation via CLIP
# 5. BRISQUE:  Classic Image Quality Metric
# 6. FID:      Frechet Inception Distance
##########################################################

# Arguments
parser = ArgumentParser()
parser.add_argument('--generated_image', type=str, default='')
parser.add_argument('--clean_image',  type=str, default='')
parser.add_argument('--log_dir',         type=str, default='')
parser.add_argument('--size',            type=int, default=512)
args = parser.parse_args()


# CLIP model
clip_model, clip_image_preprocess = clip.load('ViT-B/32', 'cuda')
clip_text_preprocess = lambda text : clip.tokenize(text).detach().clone().cuda()

# Load training images
train_images = []
for dir in os.listdir(args.clean_image):
    img = Image.open(os.path.join(args.clean_image, dir))
    train_images.append(img)

# Load generated images
generate_images = []
for img in os.listdir(args.generated_image):
    if 'png' in img or 'jpg' in img or 'jpeg' in img:
        image_path = os.path.join(args.generated_image, img)
        generate_images.append(Image.open(image_path))


with torch.no_grad():
    
    # Face Detection Similarity
    print('Evaluating FDS')
    mtcnn = MTCNN(image_size=args.size, margin=0, device='cuda')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    train_face_embedding, generate_face_embedding = [], []
    for image in train_images:
        latent = mtcnn(image) 
        train_face_embedding.append(resnet(latent.unsqueeze(0).cuda()) if latent is not None else torch.zeros((1, 512)).cuda())
    for image in generate_images:
        latent = mtcnn(image) 
        generate_face_embedding.append(resnet(latent.unsqueeze(0).cuda()) if latent is not None else torch.zeros((1, 512)).cuda())

    resnet = resnet.cpu()
    mtcnn = mtcnn.cpu()
    generate_face_embedding = torch.stack(generate_face_embedding, dim=0)
    FDS_score = torch.tensor([torch.cosine_similarity(train_face_embed.cuda(), generate_face_embedding.cuda()).mean()
                        for train_face_embed in train_face_embedding]).mean().cpu().item()


    # SSIM & PSNR
    print('Evaluating SSIM and PSNR')
    ssim_score, psnr_score = 0, 0
    transform = transforms.Compose([transforms.Resize((args.size, args.size)), transforms.ToTensor()])
    transform_train_image = torch.stack([transform(img) for img in train_images])
    for image in generate_images:
        transform_generate_image = transform(image)
        ssim_score += ssim(transform_train_image, transform_generate_image.unsqueeze(0).expand(transform_train_image.shape[0], -1, -1, -1)).cpu().mean().item()
        psnr_score += psnr(transform_train_image, transform_generate_image.unsqueeze(0).expand(transform_train_image.shape[0], -1, -1, -1)).cpu().mean().item()
    ssim_score /= len(generate_images)
    psnr_score /= len(generate_images)


    # CLIP Image Quality Score 
    print('Evaluating IQS')
    IQS_image_embedding = []
    for image in generate_images:
        prepross_image = clip_image_preprocess(image)
        with torch.no_grad():
            IQS_image_embedding.append(clip_model.encode_image(prepross_image.unsqueeze(0).cuda()))
    IQS_image_embedding = torch.cat(IQS_image_embedding, dim=0)
    good_prompt = clip_text_preprocess('A good photo of high quality')
    bad_prompt = clip_text_preprocess('A bad photo of low quality')
    good_prompt = clip_model.encode_text(good_prompt)
    bad_prompt = clip_model.encode_text(bad_prompt)
    clip_model = clip_model.cpu()
    IQS_score = 1000 * torch.mean(torch.cosine_similarity(IQS_image_embedding, good_prompt) - torch.cosine_similarity(IQS_image_embedding, bad_prompt))

    # Evaluate FID
    metrics_dict = fid.calculate_metrics(
        input1=args.clean_image,
        input2=args.generated_image,
        cuda=True,
        fid=True,
        verbose=False,
    )
    FID_score = metrics_dict['frechet_inception_distance']
    
    # Save the evaluation results
    with open(f'{args.log_dir}/log.txt', 'w') as f:
        f.write(f'Path = {args.log_dir}\n')
        f.write(f'FDS = {FDS_score}, SSIM={ssim_score}, PSNR={psnr_score}, IQS={IQS_score}, FID = {FID_score}\n')

    # Evaluate BRISQUE
    print('Evaluating BRISQUE')
    os.system(f'cd ./libsvm/python; python brisquequality.py ../../{args.generated_image} {args.log_dir}')

