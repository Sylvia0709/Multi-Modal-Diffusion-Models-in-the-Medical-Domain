import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tab_transformer_pytorch import TabTransformer

def main():
    print("TabTransformer Simple Test")
    
    # Load mutation.csv file
    try:
        data_path = "dataset/mutation.csv"
        print(f"Attempting to load file: {data_path}")
        df = pd.read_csv(data_path)
        print(f"File loaded successfully! Dataset shape: {df.shape}")
        
        # Show first few rows and column names
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        print("\nColumn names:")
        print(df.columns.tolist())
        
        # Check data types
        print("\nData types:")
        print(df.dtypes)
        
        # Automatically identify categorical and numerical features
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        print(f"\nIdentified categorical features ({len(cat_cols)}):")
        print(cat_cols)
        
        print(f"\nIdentified numerical features ({len(num_cols)}):")
        print(num_cols)
        
        # Assume last column is the target variable
        target_col = num_cols[-1]  # Assume the last numerical column is the target
        num_cols = num_cols[:-1]  # Remove target from features
        
        print(f"\nAssumed target variable: {target_col}")
        
        # Process categorical features - convert to integer encoding
        encoders = {}
        encoded_df = df.copy()
        
        for col in cat_cols:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
        # Process numerical features - standardize
        scaler = StandardScaler()
        encoded_df[num_cols] = scaler.fit_transform(encoded_df[num_cols])
        
        # Prepare inputs and target
        X_cat = encoded_df[cat_cols].values if cat_cols else np.zeros((len(encoded_df), 0))
        X_num = encoded_df[num_cols].values if num_cols else np.zeros((len(encoded_df), 0))
        y = encoded_df[target_col].values
        
        # Split into train and test sets
        X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
            X_cat, X_num, y, test_size=0.2, random_state=42
        )
        
        # Get the number of categories for each categorical feature
        categories = []
        for col in cat_cols:
            categories.append(encoded_df[col].nunique())
            
        if not categories:  # If no categorical features, add a dummy
            categories = [1]
            X_cat_train = np.zeros((len(X_num_train), 1), dtype=int)
            X_cat_test = np.zeros((len(X_num_test), 1), dtype=int)
            
        # Create model
        print("\nInitializing TabTransformer model...")
        model = TabTransformer(
            categories=categories,          # Number of categories for each feature
            num_continuous=X_num.shape[1],  # Number of continuous features
            dim=32,                         # Embedding dimension
            depth=3,                        # Number of transformer layers
            heads=4,                        # Number of attention heads
            attn_dropout=0.1,               # Attention dropout rate
            ff_dropout=0.1,                 # Feed-forward dropout rate
            dim_out=1                       # Output dimension (1 for regression)
        )
        
        # Convert to PyTorch tensors
        X_cat_train_tensor = torch.LongTensor(X_cat_train)
        X_num_train_tensor = torch.FloatTensor(X_num_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        X_cat_test_tensor = torch.LongTensor(X_cat_test)
        X_num_test_tensor = torch.FloatTensor(X_num_test)
        
        # Test model forward pass
        print("\nTesting model forward pass...")
        model.eval()
        with torch.no_grad():
            # Test with a small batch
            batch_size = min(5, len(X_cat_train))
            outputs = model(
                X_cat_train_tensor[:batch_size], 
                X_num_train_tensor[:batch_size]
            )
            
            print(f"Input batch shapes: Categorical={X_cat_train_tensor[:batch_size].shape}, Numerical={X_num_train_tensor[:batch_size].shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Output values: {outputs.squeeze().tolist()}")
            
        print("\nShort training test (5 batches)...")
        # Set up optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss() if len(np.unique(y)) > 10 else torch.nn.BCEWithLogitsLoss()
        
        model.train()
        batch_size = 32
        
        # Simple test - train for a few batches
        for i in range(5):
            # Get random batch
            indices = np.random.choice(len(X_cat_train), batch_size)
            x_cat_batch = X_cat_train_tensor[indices]
            x_num_batch = X_num_train_tensor[indices]
            y_batch = y_train_tensor[indices]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x_cat_batch, x_num_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"Batch {i+1}, Loss: {loss.item():.6f}")
        
        print("\nTabTransformer test completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()