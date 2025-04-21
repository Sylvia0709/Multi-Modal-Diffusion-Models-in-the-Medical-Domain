
from os.path import split, join
import argparse
import logging
import os
import random
import sys
import h5py

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/test_vol_h5',
                    help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='datasets', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, default='./features', help='output dir for features')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--feature_size', type=int, default=768, help='dimension of extracted features')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces')
parser.add_argument('--patch_size', type=list, default=[224, 224], help='patch size to extract features')
parser.add_argument('--pool_method', type=str, default='avg', choices=['avg', 'max', 'none'],
                    help='pooling method for feature extraction: avg, max, or none (return feature maps)')
parser.add_argument("--split_name", default="test", help="Directory of the input list")

args = parser.parse_args()
config = get_config(args)



class FeatureExtractionDataset(Dataset):
    def __init__(self, base_dir, list_dir, split):
        self.base_dir = base_dir
        self.sample_list = []
        

        if os.path.exists(list_dir):
            with open(join(list_dir, f'{split}.txt'), 'r') as f:
                self.sample_list = [line.strip() for line in f]
        else:
           
            valid_extensions = ['.npz', '.h5', '.nii.gz']
            for file in os.listdir(base_dir):
                if any(file.endswith(ext) for ext in valid_extensions):
                    self.sample_list.append(file)
        
        print(f"Total number of samples: {len(self.sample_list)}")
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case_name = self.sample_list[idx]
        
 
        if case_name.endswith('.npz'):
            data = np.load(join(self.base_dir, case_name), allow_pickle=True)
   
            image = None
            for key_name in ['image', 'data', 'img', 'vol']:
                if key_name in data:
                    image = data[key_name]
                    break
            if image is None:
                image = data[list(data.keys())[0]]
                
        elif case_name.endswith('.h5'):
            data = h5py.File(join(self.base_dir, case_name), 'r')
            image = data['image'][:]
            
        else:  
            try:
                from PIL import Image
                img = Image.open(join(self.base_dir, case_name))
                image = np.array(img)
            except:
                print(f"Unsupported file format: {case_name}")
                image = np.zeros((224, 224, 3))  
                

        if len(image.shape) == 2:  
            image = np.expand_dims(image, axis=0)  
        elif len(image.shape) == 3 and image.shape[2] <= 4: 
            image = np.transpose(image, (2, 0, 1))  
        elif len(image.shape) == 3:  

            mid_slice = image.shape[0] // 2
            image = image[mid_slice:mid_slice+1]  
        elif len(image.shape) == 4:  
            if image.shape[0] <= 4:  
                mid_slice = image.shape[1] // 2
                image = image[:, mid_slice:mid_slice+1]  
            else:  
                mid_slice = image.shape[0] // 2
                image = np.transpose(image[mid_slice], (2, 0, 1)) 
        

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
            

        image_tensor = torch.from_numpy(image)
        if image_tensor.shape[-1] != args.img_size or image_tensor.shape[-2] != args.img_size:

            if len(image_tensor.shape) == 3:  # [C,H,W]
                from torchvision import transforms
                resize = transforms.Resize((args.img_size, args.img_size))
                image_tensor = resize(image_tensor)
            elif len(image_tensor.shape) == 4:  # [C,D,H,W]
                from torchvision import transforms
                resize = transforms.Resize((args.img_size, args.img_size))
    
                slices = []
                for i in range(image_tensor.shape[1]):
                    slice_i = image_tensor[:, i, :, :]
                    slice_i = resize(slice_i)
                    slices.append(slice_i.unsqueeze(1))
                image_tensor = torch.cat(slices, dim=1)
                

        if len(image_tensor.shape) == 3: 
            image_tensor = image_tensor.unsqueeze(0)  
            
        return {
            "image": image_tensor,
            "case_name": case_name
        }



class SwinEncoder(torch.nn.Module):
    def __init__(self, model):
        super(SwinEncoder, self).__init__()

        self.swinViT = model.swinViT
        
    def forward(self, x):

        encoder_outputs = self.swinViT(x)
        

        if args.pool_method == 'avg':

            features = []
            for feat in encoder_outputs:

                dims = tuple(range(2, len(feat.shape)))
                pooled = torch.mean(feat, dim=dims)
                features.append(pooled)
            return features
        
        elif args.pool_method == 'max':

            features = []
            for feat in encoder_outputs:
                dims = tuple(range(2, len(feat.shape)))
                pooled = torch.amax(feat, dim=dims)
                features.append(pooled)
            return features
        
        else:

            return encoder_outputs



def extract_features(args, model):

    db_test = FeatureExtractionDataset(
        base_dir=args.root_path, 
        list_dir=args.list_dir,
        split=args.split_name
    )
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logging.info(f"Total number of samples: {len(db_test)}")
    

    os.makedirs(args.output_dir, exist_ok=True)
    

    model.eval()
    

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, case_name = sampled_batch["image"], sampled_batch["case_name"][0]
            
  
            if args.dataset == "datasets":
                case_name = split(case_name.split(",")[0])[-1]
            
     
            image = image.cuda()
            
    
            features = model(image)
            
    
            feature_path = join(args.output_dir, f"{os.path.splitext(case_name)[0]}_features.npz")
            
 
            if isinstance(features, list):
                features_np = [feat.cpu().numpy() for feat in features]

                np.savez(feature_path, *features_np)
                logging.info(f"Saved multi-level features for {case_name} with shapes: {[feat.shape for feat in features_np]}")
            else:

                features_np = features.cpu().numpy()
                np.save(feature_path, features_np)
                logging.info(f"Saved features for {case_name} with shape: {features_np.shape}")
            
    return "Feature extraction finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    dataset_name = args.dataset
    dataset_config = {
        args.dataset: {
            'root_path': args.root_path,
            'list_dir': f'./lists/{args.dataset}',
        },
    }
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    
 
    log_folder = './feature_extraction_log/'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/feature_extraction.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

  
    full_model = ViT_seg(config, img_size=args.img_size, num_classes=1).cuda()
    
    if hasattr(args, 'output_dir') and args.output_dir:
        snapshot = os.path.join(args.output_dir, 'best_model.pth')
        if not os.path.exists(snapshot):
            snapshot = os.path.join(args.output_dir, 'epoch_last.pth')
        
        if os.path.exists(snapshot):
            checkpoint = torch.load(snapshot, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                full_model.load_state_dict(checkpoint['model'])
                logging.info(f"Loaded weights from {snapshot} (model key)")
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                full_model.load_state_dict(checkpoint['state_dict'])
                logging.info(f"Loaded weights from {snapshot} (state_dict key)")
            else:
                full_model.load_state_dict(checkpoint)
                logging.info(f"Loaded weights from {snapshot}")
            
            logging.info(f"Loaded pretrained model from {snapshot}")
        else:
            logging.info("No pretrained weights found, using random initialization")
    
  
    encoder_model = SwinEncoder(full_model)
    encoder_model.cuda()
    
    
    extract_features(args, encoder_model)
    
    logging.info("Feature extraction complete!")