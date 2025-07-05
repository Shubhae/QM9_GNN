# SIMPLE GNN MODEL FOR QM9 - WORKING VERSION
# Simplified Graph Neural Network for molecular property prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("RDKit not available. Some molecular descriptors will be skipped.")
    RDKIT_AVAILABLE = False

class SimpleMolecularGNN(nn.Module):
    """
    Simple Graph Neural Network for molecular property prediction.
    """
    
    def __init__(self, num_node_features, hidden_channels=32, num_layers=2):
        super(SimpleMolecularGNN, self).__init__()
        
        self.num_layers = num_layers
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Global pooling
        self.global_pool = global_mean_pool
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, batch):
        # Graph convolutional layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling
        x = self.global_pool(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        
        return x.squeeze()

class SimpleQM9GNNPipeline:
    """
    Simple QM9 pipeline using Graph Neural Networks.
    """
    
    def __init__(self, data_path='qm9_subset.csv'):
        self.data_path = data_path
        self.data = None
        self.cleaned_data = None
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_data(self):
        """Load the QM9 dataset."""
        print("🔍 Loading QM9 dataset...")
        
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded! Shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def clean_data(self):
        """Clean the dataset."""
        print("\n🧹 Cleaning dataset...")
        
        if self.data is None:
            print("❌ No data loaded!")
            return False
        
        # Make a copy
        self.cleaned_data = self.data.copy()
        
        # Remove missing values in target
        target_col = 'gap'
        initial_rows = len(self.cleaned_data)
        self.cleaned_data = self.cleaned_data.dropna(subset=[target_col])
        removed_rows = initial_rows - len(self.cleaned_data)
        print(f"📉 Removed {removed_rows} rows with missing target")
        
        # Handle outliers in target
        Q1 = self.cleaned_data[target_col].quantile(0.25)
        Q3 = self.cleaned_data[target_col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (self.cleaned_data[target_col] < Q1 - 1.5*IQR) | (self.cleaned_data[target_col] > Q3 + 1.5*IQR)
        outliers_removed = outlier_mask.sum()
        self.cleaned_data = self.cleaned_data[~outlier_mask]
        print(f"📉 Removed {outliers_removed} outliers from target")
        
        print(f"✅ Data cleaned! Final shape: {self.cleaned_data.shape}")
        return True
    
    def smiles_to_graph(self, smiles):
        """Convert SMILES to molecular graph."""
        if not RDKIT_AVAILABLE:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Get atom features (simplified)
            atom_features = []
            for atom in mol.GetAtoms():
                # Simple atom features
                atom_features.append([
                    atom.GetAtomicNum(),  # Atomic number
                    atom.GetDegree(),     # Degree
                    atom.GetImplicitValence(),  # Implicit valence
                    atom.GetFormalCharge(),     # Formal charge
                    atom.GetIsAromatic(),       # Aromaticity
                    atom.GetHybridization(),    # Hybridization
                    atom.GetTotalNumHs(),       # Total H count
                ])
            
            # Get edge indices
            edge_index = []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_index.append([start, end])
                edge_index.append([end, start])  # Undirected graph
            
            if len(edge_index) == 0:
                return None
            
            # Convert to tensors
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            return Data(x=x, edge_index=edge_index)
            
        except:
            return None
    
    def create_graph_dataset(self, smiles_list, labels):
        """Create graph dataset from SMILES."""
        print("🕸️ Creating molecular graphs...")
        
        graphs = []
        valid_labels = []
        
        for i, (smiles, label) in enumerate(zip(smiles_list, labels)):
            if i % 100 == 0:
                print(f"   Processing molecule {i+1}/{len(smiles_list)}")
            
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                graph.y = torch.tensor([label], dtype=torch.float)
                graphs.append(graph)
                valid_labels.append(label)
        
        print(f"✅ Created {len(graphs)} valid molecular graphs")
        return graphs, valid_labels
    
    def train_gnn_model(self, train_graphs, val_graphs, epochs=50):
        """Train the GNN model."""
        print(f"\n🌲 Training GNN model...")
        
        if len(train_graphs) == 0 or len(val_graphs) == 0:
            print("❌ No valid graphs for training!")
            return None, None
        
        # Create data loaders
        train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
        
        # Initialize model
        num_node_features = train_graphs[0].x.size(1)
        self.model = SimpleMolecularGNN(num_node_features=num_node_features).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item()
                    
                    val_preds.extend(out.cpu().numpy())
                    val_targets.extend(batch.y.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_r2 = r2_score(val_targets, val_preds)
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        print("✅ GNN model trained!")
        return val_preds, val_targets
    
    def run_pipeline(self):
        """Run the complete GNN pipeline."""
        print("🚀 Starting Simple QM9 GNN Pipeline")
        print("=" * 50)
        
        # Load data
        if not self.load_data():
            return False
        
        # Clean data
        if not self.clean_data():
            return False
        
        # Split data
        X = self.cleaned_data[['smiles']]
        y = self.cleaned_data['gap']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"📊 Training set: {len(X_train)}")
        print(f"📊 Test set: {len(X_test)}")
        
        # Create graph datasets
        train_graphs, train_labels = self.create_graph_dataset(
            X_train['smiles'].values, y_train.values
        )
        test_graphs, test_labels = self.create_graph_dataset(
            X_test['smiles'].values, y_test.values
        )
        
        if len(train_graphs) == 0 or len(test_graphs) == 0:
            print("❌ Could not create valid molecular graphs!")
            return False
        
        # Train GNN model
        val_preds, val_targets = self.train_gnn_model(train_graphs, test_graphs, epochs=30)
        
        if val_preds is None:
            print("❌ GNN training failed!")
            return False
        
        # Calculate metrics
        test_mae = mean_absolute_error(val_targets, val_preds)
        test_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        test_r2 = r2_score(val_targets, val_preds)
        
        print("\n📈 GNN Model Performance:")
        print(f"   Test MAE: {test_mae:.4f}")
        print(f"   Test RMSE: {test_rmse:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        
        # Compare with baseline
        baseline_r2 = -0.1249
        improvement = test_r2 - baseline_r2
        print(f"\n📈 Improvement:")
        print(f"   Baseline R²: {baseline_r2:.4f}")
        print(f"   GNN R²: {test_r2:.4f}")
        print(f"   Improvement: {improvement:.4f}")
        
        if improvement > 0:
            print("   ✅ GNN model improved performance!")
        else:
            print("   ❌ GNN model did not improve")
        
        # Create plots
        self.create_plots(val_preds, val_targets)
        
        # Save model
        self.save_model()
        
        print("\n🎉 Simple GNN Pipeline completed!")
        print("=" * 50)
        
        return {
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'improvement': improvement
        }
    
    def create_plots(self, y_pred, y_true):
        """Create evaluation plots."""
        print("📊 Creating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([min(y_true), max(y_true)], 
                       [min(y_true), max(y_true)], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual HOMO-LUMO Gap')
        axes[0, 0].set_ylabel('Predicted HOMO-LUMO Gap')
        axes[0, 0].set_title('GNN: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = np.array(y_true) - np.array(y_pred)
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('GNN: Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('GNN: Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Target distribution
        axes[1, 1].hist(y_true, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('HOMO-LUMO Gap')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('GNN: Target Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qm9_simple_gnn_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ GNN plots saved as 'qm9_simple_gnn_evaluation.png'")
    
    def save_model(self, model_path='qm9_simple_gnn_model.pth'):
        """Save the GNN model."""
        print(f"\n💾 Saving GNN model to {model_path}...")
        
        if self.model is None:
            print("❌ No model to save!")
            return False
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_info': {
                'target_property': 'HOMO-LUMO Gap',
                'dataset': 'QM9',
                'model_type': 'Simple Graph Neural Network',
                'architecture': 'GCN + Global Pooling + FC',
                'improvements': [
                    'Graph Neural Network architecture',
                    'Molecular graph representation',
                    'Graph convolutional layers',
                    'Global pooling',
                    'Simplified atom features'
                ]
            }
        }, model_path)
        
        print(f"✅ GNN model saved!")
        return True

def main():
    """Main function."""
    print("🔬 QM9 HOMO-LUMO Gap Prediction with Simple GNN")
    print("=" * 50)
    
    # Check if PyTorch Geometric is available
    try:
        import torch_geometric
        print("✅ PyTorch Geometric is available")
    except ImportError:
        print("❌ PyTorch Geometric not available.")
        return
    
    model = SimpleQM9GNNPipeline()
    results = model.run_pipeline()
    
    if results:
        print(f"\n📋 Final GNN Results:")
        print(f"   Test MAE: {results['test_mae']:.4f}")
        print(f"   Test RMSE: {results['test_rmse']:.4f}")
        print(f"   Test R²: {results['test_r2']:.4f}")
        print(f"   Improvement: {results['improvement']:.4f}")
        
        if results['improvement'] > 0:
            print(f"\n✅ SUCCESS: GNN model improved performance!")
            print(f"🎯 Graph Neural Networks show promise for molecular property prediction")
        else:
            print(f"\n⚠️ GNN model needs further tuning")
            print(f"💡 Consider: more data, different GNN architectures, hyperparameter tuning")

if __name__ == "__main__":
    main() 