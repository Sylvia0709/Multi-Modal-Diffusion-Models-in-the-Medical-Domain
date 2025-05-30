import os
# solve possiable OMP/MKL problem
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
import warnings
from torch.utils.data.dataloader import default_collate

# MONAI
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Resized,
    EnsureTyped
)


# overlook useless warnging
warnings.filterwarnings("ignore", category=UserWarning, module="monai.networks.nets.swin_unetr")
warnings.filterwarnings("ignore", category=FutureWarning, module="monai.networks.nets.swin_unetr")
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension") 

# Lora part
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRALinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_up(self.lora_down(x))

class WindowAttentionWithLoRA(nn.Module):
    def __init__(self, dim, num_heads, rank=4):
        super(WindowAttentionWithLoRA, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = LoRALinear(dim, dim * 3, rank=rank) 
        self.proj = nn.Linear(dim, dim)



    def forward(self, x, mask=None):
        B_, N, C = x.shape 
        qkv = self.qkv(x).view(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        if mask is not None:
            nW = mask.shape[0] 
            attn = attn.view(-1, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N) 
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinUNETRWithWindowLoRA(nn.Module):

    def __init__(self, img_size, in_channels, out_channels, feature_size, rank=4, use_checkpoint=True):
        super(SwinUNETRWithWindowLoRA, self).__init__()
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint
        )
        self.rank = rank
        self._replace_window_attention()

    def _replace_window_attention(self):
        replaced_count = 0
        for name, module in self.swin_unetr.named_modules():
            parent_name_parts = name.split('.')
            if len(parent_name_parts) > 1 and parent_name_parts[-1] == 'attn':
                 parent_name = '.'.join(parent_name_parts[:-1])
                 parent_module = self.swin_unetr.get_submodule(parent_name)
                 if hasattr(parent_module, 'attn') and getattr(parent_module, 'attn') is module:
                     if hasattr(module, "num_heads") and hasattr(module, "qkv") and isinstance(getattr(module, 'qkv', None), nn.Linear):
                         original_qkv = module.qkv
                         dim = original_qkv.in_features
                         num_heads = module.num_heads
                         new_attn = WindowAttentionWithLoRA(dim=dim, num_heads=num_heads, rank=self.rank)
                         setattr(parent_module, 'attn', new_attn)
                         print(f"Replaced attention at: {name} with LoRA version (dim={dim}, heads={num_heads})")
                         replaced_count += 1
        if replaced_count == 0:
             print("Warning: No WindowAttention modules were replaced. Check the replacement logic and MONAI version.")


    def forward(self, x):
        return self.swin_unetr(x)

    def get_encoder_features(self, x):
        if hasattr(self.swin_unetr, 'swinViT'):
            encoder_output = self.swin_unetr.swinViT(x)
            if isinstance(encoder_output, (list, tuple)):
                return encoder_output[-1]
            else:
                return encoder_output
        else:
            raise AttributeError("SwinUNETR instance does not have 'swinViT' attribute.")


class KIRC_CT_Dataset(Dataset):
    def __init__(self, data_dir, target_img_size):

        self.data_dir = data_dir
        self.file_paths = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npz')])
        self.target_img_size = target_img_size

        self.transforms = Compose(
            [
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"), 
                Resized(keys=["image"],
                        spatial_size=self.target_img_size,
                        mode='trilinear',
                        align_corners=False),
                EnsureTyped(keys=["image"], dtype=torch.float32),
            ]
        )
        print(f"Dataset initialized. Target image size set to: {self.target_img_size}")
        print(f"Applied MONAI transforms: EnsureChannelFirstd -> Resized -> EnsureTyped")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image_np = None
        original_shape_str = "N/A" 
        try:
            with np.load(file_path) as data:
                if "image" in data:
                    image_np = data["image"]
                elif "vol_data" in data:
                    image_np = data["vol_data"]
                elif data.files:
                    first_key = data.files[0]
                    image_np = data[first_key]
                else:
                     raise ValueError(f"No data arrays found in file {file_path}")

                original_shape_str = str(image_np.shape) 

                print(f"DEBUG: Processing {os.path.basename(file_path)} - Original shape: {image_np.shape}, dtype: {image_np.dtype}")

                if not isinstance(image_np, np.ndarray):
                    raise TypeError(f"Loaded 'image' is not a numpy array, type is {type(image_np)}")
                if not np.issubdtype(image_np.dtype, np.number):
                     raise TypeError(f"Loaded 'image' dtype is non-numeric ({image_np.dtype})")


                if image_np.ndim == 4:
                    if image_np.shape[0] == 1: 
                        print(f"DEBUG: Input is 4D with shape {image_np.shape}. Squeezing axis 0.")
                        image_np = np.squeeze(image_np, axis=0) 
                        print(f"DEBUG: Shape after squeeze: {image_np.shape}")
                        if image_np.shape[-1] == min(image_np.shape): 
                             print(f"DEBUG: Assuming shape is (H, W, D). Permuting to (D, H, W).")
                             image_np = np.transpose(image_np, (2, 0, 1)) 
                             print(f"DEBUG: Shape after permute: {image_np.shape}")
                        elif image_np.shape[0] == min(image_np.shape):
                             print(f"DEBUG: Assuming shape is already (D, H, W) after squeeze.")
                        else:
                             print(f"WARNING: Shape after squeeze is {image_np.shape}. Assuming it's (D, H, W) for MONAI.")

                    else: 
                         raise ValueError(f"Loaded image has 4 dimensions, but the first is not 1: {image_np.shape}. Cannot automatically handle.")
                elif image_np.ndim == 3:
                    print(f"DEBUG: Input is 3D with shape {image_np.shape}.")
  
                    if image_np.shape[-1] == min(image_np.shape): 
                         print(f"DEBUG: Assuming shape is (H, W, D). Permuting to (D, H, W).")
                         image_np = np.transpose(image_np, (2, 0, 1))
                         print(f"DEBUG: Shape after permute: {image_np.shape}")
                    elif image_np.shape[0] == min(image_np.shape):
                         print(f"DEBUG: Assuming shape is already (D, H, W).")
                    else:
                         print(f"WARNING: Shape is {image_np.shape}. Assuming it's (D, H, W) for MONAI.")
                else:
                    raise ValueError(f"Loaded image has unexpected number of dimensions: {image_np.ndim} (shape: {image_np.shape})")

            data_dict = {"image": image_np}

            transformed_dict = self.transforms(data_dict) 

            image_tensor = transformed_dict["image"]

            return image_tensor, None, file_path

        except Exception as e:
            import traceback
            print(f"ERROR processing file {file_path}:")
            print(f"  Original np array shape: {original_shape_str}") 
            if image_np is not None:
                 print(f"  np array shape before MONAI transforms: {image_np.shape}")
            else:
                 print("  Input np array was not loaded successfully or failed before shape logging.")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {e}")
            print("  Traceback:")
            traceback.print_exc()
            print("-" * 30)
            return None, None, file_path


captured_features = {}

def get_features_hook(module, input, output):
    feature = output[-1] if isinstance(output, (list, tuple)) else output
    captured_features['last_feature'] = feature.detach().cpu()

def extract_features_with_hook(data_loader, model, device, feature_save_dir):
    model.eval()
    hook_handle = None
    os.makedirs(feature_save_dir, exist_ok=True)

    try:
        target_module = model.swin_unetr.swinViT
        hook_handle = target_module.register_forward_hook(get_features_hook)
        print(f"Registered forward hook on {type(target_module).__name__}")
    except AttributeError:
        print("ERROR: Could not find 'model.swin_unetr.swinViT'. Check model structure.")
        return {}

    all_features_paths = {}
    processed_batches = 0
    total_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, collated_data in enumerate(data_loader):
            if collated_data is None:
                print(f"Skipping empty or problematic batch {batch_idx+1}/{total_batches} (collate returned None).")
                continue

            images_tensor, _, file_paths = collated_data 

            if images_tensor is None:
                 print(f"Skipping batch {batch_idx+1}/{total_batches} due to None image tensor after collation.")
                 continue

            images_tensor = images_tensor.to(device)
            captured_features.clear()

            try:
                _ = model(images_tensor)
            except Exception as e:
                print(f"ERROR during model forward pass for batch {batch_idx+1}/{total_batches}: {e}")
                print(f"Input tensor shape: {images_tensor.shape}, dtype: {images_tensor.dtype}, device: {images_tensor.device}")
                continue 

            if 'last_feature' in captured_features:
                batch_features = captured_features['last_feature']
                if batch_features.shape[0] != len(file_paths):
                     print(f"Warning: Mismatch between feature batch size ({batch_features.shape[0]}) and number of file paths ({len(file_paths)}) in batch {batch_idx+1}/{total_batches}. Skipping saving.")
                     continue

                for i in range(batch_features.shape[0]):
                    try:
                        original_filename = os.path.basename(file_paths[i])
                        feature_filename = original_filename.replace(".npz", "_features.pt")
                        save_path = os.path.join(feature_save_dir, feature_filename)
                        sample_feature = batch_features[i]
                        torch.save(sample_feature, save_path)
                        all_features_paths[original_filename] = save_path
                    except IndexError:
                         print(f"Error accessing file_paths[{i}] for batch {batch_idx+1}/{total_batches}. Skipping save.")
                    except Exception as e:
                         print(f"Error saving feature for {file_paths[i] if i < len(file_paths) else 'unknown file'} in batch {batch_idx+1}/{total_batches}: {e}")
            else:
                print(f"Warning: Hook did not capture features for batch {batch_idx+1}/{total_batches}.")

            processed_batches += 1
            if processed_batches % 10 == 0 or processed_batches == total_batches:
                 print(f"Processed batch {processed_batches}/{total_batches}")

    if hook_handle:
        hook_handle.remove()
        print("Removed forward hook.")
    else:
        print("Warning: Hook handle was not valid, couldn't remove hook.")

    return all_features_paths



if __name__ == "__main__":


    data_dir = "./datasets/kirc_ct_npz"
    feature_save_dir = "./extracted_features_resized" 

    img_size = (96, 96, 96)
    in_channels = 1      
    out_channels = 3     
    feature_size = 48    
    batch_size = 4       
    rank = 4             
    use_checkpoint = False 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Initializing dataset from: {data_dir}")
 
    try:
        dataset = KIRC_CT_Dataset(data_dir, target_img_size=img_size)
    except Exception as e:
        print(f"ERROR initializing dataset: {e}")
        exit()

    if len(dataset) == 0:
        print(f"ERROR: No .npz files found or processed in {data_dir}. Exiting.")
        exit()
    else:
        print(f"Found {len(dataset)} samples in {data_dir}.")

    def safe_collate_fn(batch):

        valid_batch = [item for item in batch if item[0] is not None]
        if not valid_batch:
            return None 

        images = [item[0] for item in valid_batch]
        paths = [item[2] for item in valid_batch]

        try:
            
            collated_images = default_collate(images)
        except RuntimeError as e:
            
            print(f"ERROR during image collation: {e}")
            print("Shapes in this batch:")
            for i, img in enumerate(images):
                 print(f"  Item {i} (Path: {paths[i]}): {img.shape}")
            return None 
        except Exception as e:
            print(f"Unexpected ERROR during image collation: {e}")
            return None
        return collated_images, None, paths

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,   
        collate_fn=safe_collate_fn, 
        pin_memory=False 
    )
    print("DataLoader initialized.")

    print("Initializing model...")
    try:

        model = SwinUNETRWithWindowLoRA(
            img_size=img_size, 
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            rank=rank,
            use_checkpoint=use_checkpoint
        )
        model = model.to(device)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"ERROR initializing model: {e}")
        exit()

    print("\nStarting feature extraction...")
    extracted_feature_paths = extract_features_with_hook(data_loader, model, device, feature_save_dir)

    print("\nFeature extraction complete.")
    if extracted_feature_paths:
        print(f"Extracted {len(extracted_feature_paths)} features saved in: {feature_save_dir}")

        first_key = list(extracted_feature_paths.keys())[0]
        first_feature_path = extracted_feature_paths[first_key]
        try:
            loaded_feature = torch.load(first_feature_path, map_location='cpu')
            print(f"\nSuccessfully loaded example feature from {first_feature_path}")
            print(f"Feature shape: {loaded_feature.shape}")
            print(f"Feature dtype: {loaded_feature.dtype}")
        except Exception as e:
            print(f"\nError loading example feature from {first_feature_path}: {e}")
    else:
        print("No features were extracted or saved.")