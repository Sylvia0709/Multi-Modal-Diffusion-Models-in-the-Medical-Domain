
import pandas as pd
import torch
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tab_transformer_pytorch import TabTransformer
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def extract_tab_features(csv_path, feature_dim=256):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"path error: {csv_path}")
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError("feature_dim error")
    

    
    file_name = os.path.splitext(os.path.basename(csv_path))[0]
    

    try:
        df = pd.read_csv(csv_path)
        logging.info(f"datashape: {df.shape}")
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        

        
  
        encoders = {}
        encoded_df = df.copy()
        for col in cat_cols:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
        scaler = StandardScaler()
        if num_cols: 
            if encoded_df[num_cols].isnull().values.any():
                encoded_df[num_cols] = encoded_df[num_cols].fillna(encoded_df[num_cols].mean())
            encoded_df[num_cols] = scaler.fit_transform(encoded_df[num_cols])
        
   
        X_cat = encoded_df[cat_cols].values if cat_cols else np.zeros((len(encoded_df), 0))
        X_num = encoded_df[num_cols].values if num_cols else np.zeros((len(encoded_df), 0))
        
  
        categories = [encoded_df[col].nunique() for col in cat_cols]
        if not categories:  
            categories = [1]
            X_cat = np.zeros((len(encoded_df), 1), dtype=int)
        

        dim = max(feature_dim // 4, 8)  
        model = TabTransformer(
            categories=categories,
            num_continuous=X_num.shape[1],
            dim=dim,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            dim_out=feature_dim
        )
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logging.info(f"model loaded in : {device}")
        
 
        X_cat_tensor = torch.LongTensor(X_cat).to(device)
        X_num_tensor = torch.FloatTensor(X_num).to(device)
        

        model.eval()
        batch_size = 64 
        num_samples = len(encoded_df)
        all_features = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                cat_batch = X_cat_tensor[i:end_idx]
                num_batch = X_num_tensor[i:end_idx]
                

                features = model(cat_batch, num_batch).cpu() 
                all_features.append(features.numpy())

                if (i // batch_size) % 10 == 0:
                    logging.info(f"process: {end_idx}/{num_samples}")
        

        all_features = np.vstack(all_features)
        
        logging.info(f"features.shape: {all_features.shape}")
        output_path = os.path.join("result", f"{file_name}_features.npy")
        np.save(output_path, all_features)
 
        print("features:")
        print(all_features)
        return all_features
        
    except Exception as e:
        logging.error(f"error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":

    csv_path = "mutation.csv"  # dataset path
    feature_dim = 256
    
 
    features = extract_tab_features(
        csv_path=csv_path,
        feature_dim=feature_dim
    )
    
    if features is not None:
        logging.info("done!")
    else:
        logging.error("error!!!")