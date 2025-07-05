# QM9 COMPLETE AI MODEL PIPELINE - Task Group 3 Implementation
# Complete implementation with all required features and models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    from rdkit.Chem.AtomPairs import Pairs, Torsions
    from rdkit.Chem.EState import EState
    from rdkit.Chem.Fragments import fr_Al_COO, fr_Al_OH, fr_Ar_COO, fr_Ar_OH
    RDKIT_AVAILABLE = True
except ImportError:
    print("RDKit not available. Some molecular descriptors will be skipped.")
    RDKIT_AVAILABLE = False

# PyTorch and GNN imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch Geometric not available. GNN model will be skipped.")
    TORCH_AVAILABLE = False

class CoulombMatrix:
    """
    Generate Coulomb Matrix representation for molecules.
    Coulomb Matrix is a structural representation that encodes atomic interactions.
    """
    
    def __init__(self, max_atoms=50):
        self.max_atoms = max_atoms
    
    def get_coulomb_matrix(self, mol):
        """Generate Coulomb Matrix for a molecule."""
        if not RDKIT_AVAILABLE:
            return None
        
        try:
            # Get 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            conf = mol.GetConformer()
            atoms = mol.GetAtoms()
            
            n_atoms = len(atoms)
            if n_atoms > self.max_atoms:
                return None
            
            # Initialize Coulomb matrix
            cm = np.zeros((self.max_atoms, self.max_atoms))
            
            # Fill diagonal elements (nuclear charges)
            for i, atom in enumerate(atoms):
                cm[i, i] = 0.5 * atom.GetAtomicNum() ** 2.4
            
            # Fill off-diagonal elements (Coulomb interactions)
            for i in range(n_atoms):
                pos_i = conf.GetAtomPosition(i)
                for j in range(i + 1, n_atoms):
                    pos_j = conf.GetAtomPosition(j)
                    dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                    if dist > 0:
                        cm[i, j] = cm[j, i] = atoms[i].GetAtomicNum() * atoms[j].GetAtomicNum() / dist
            
            # Sort by row norm (eigenvalue sorting)
            row_norms = np.linalg.norm(cm, axis=1)
            sorted_indices = np.argsort(row_norms)[::-1]
            cm = cm[sorted_indices][:, sorted_indices]
            
            return cm.flatten()
            
        except:
            return None

class CompleteQM9Pipeline:
    """
    Complete QM9 pipeline implementing all Task Group 3 requirements.
    """
    
    def __init__(self, data_path='qm9_subset.csv'):
        self.data_path = data_path
        self.data = None
        self.cleaned_data = None
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        self.coulomb_matrix = CoulombMatrix()
        
    def load_data(self):
        """Load the QM9 dataset."""
        print("🔍 Loading QM9 dataset...")
        
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded! Shape: {self.data.shape}")
            print(f"📋 Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def clean_data(self):
        """Clean the dataset and save cleaned version."""
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
        
        # Keep essential columns
        essential_cols = ['smiles', 'gap', 'alpha', 'U0']
        available_cols = [col for col in essential_cols if col in self.cleaned_data.columns]
        self.cleaned_data = self.cleaned_data[available_cols]
        
        # Save cleaned dataset
        self.cleaned_data.to_csv('qm9_cleaned_complete.csv', index=False)
        print(f"💾 Cleaned dataset saved as 'qm9_cleaned_complete.csv'")
        
        print(f"✅ Data cleaned! Final shape: {self.cleaned_data.shape}")
        return True
    
    def compute_rdkit_descriptors(self, smiles_list):
        """Compute RDKit descriptors (15-20 features as required)."""
        if not RDKIT_AVAILABLE:
            print("⚠️ RDKit not available, skipping RDKit descriptors")
            return pd.DataFrame()
        
        print("🧪 Computing RDKit descriptors...")
        
        descriptors = []
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                print(f"   Processing molecule {i+1}/{len(smiles_list)}")
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Required descriptors (15-20 features)
                    desc = {
                        # Basic descriptors
                        'TPSA': Descriptors.TPSA(mol),
                        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
                        'NumHDonors': Descriptors.NumHDonors(mol),
                        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                        'MolWt': Descriptors.MolWt(mol),
                        'LogP': Descriptors.MolLogP(mol),
                        'NumAtoms': mol.GetNumAtoms(),
                        'NumBonds': mol.GetNumBonds(),
                        'RingCount': Descriptors.RingCount(mol),
                        'FractionCsp3': Descriptors.FractionCsp3(mol),
                        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                        
                        # Additional descriptors for better coverage
                        'MolMR': Descriptors.MolMR(mol),
                        'ExactMolWt': Descriptors.ExactMolWt(mol),
                        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
                        'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
                        'MinPartialCharge': Descriptors.MinPartialCharge(mol),
                    }
                else:
                    desc = {key: 0 for key in [
                        'TPSA', 'NumRotatableBonds', 'NumAromaticRings', 'HeavyAtomCount',
                        'NumHDonors', 'NumHAcceptors', 'MolWt', 'LogP', 'NumAtoms',
                        'NumBonds', 'RingCount', 'FractionCsp3', 'NumSaturatedRings',
                        'NumAliphaticRings', 'NumHeteroatoms', 'MolMR', 'ExactMolWt',
                        'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge'
                    ]}
                descriptors.append(desc)
            except:
                desc = {key: 0 for key in [
                    'TPSA', 'NumRotatableBonds', 'NumAromaticRings', 'HeavyAtomCount',
                    'NumHDonors', 'NumHAcceptors', 'MolWt', 'LogP', 'NumAtoms',
                    'NumBonds', 'RingCount', 'FractionCsp3', 'NumSaturatedRings',
                    'NumAliphaticRings', 'NumHeteroatoms', 'MolMR', 'ExactMolWt',
                    'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge'
                ]}
                descriptors.append(desc)
        
        return pd.DataFrame(descriptors)
    
    def compute_ecfp_fingerprints(self, smiles_list, radius=2, nBits=2048):
        """Compute ECFP (Extended Connectivity Fingerprints)."""
        if not RDKIT_AVAILABLE:
            print("⚠️ RDKit not available, skipping ECFP fingerprints")
            return pd.DataFrame()
        
        print("🔍 Computing ECFP fingerprints...")
        
        fingerprints = []
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                print(f"   Processing molecule {i+1}/{len(smiles_list)}")
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Generate Morgan fingerprints (ECFP equivalent)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                    fp_array = np.array(fp)
                    fingerprints.append(fp_array)
                else:
                    fingerprints.append(np.zeros(nBits))
            except:
                fingerprints.append(np.zeros(nBits))
        
        # Convert to DataFrame with feature names
        fp_df = pd.DataFrame(fingerprints)
        fp_df.columns = [f'ECFP_{i}' for i in range(nBits)]
        
        # Select top features (reduce dimensionality)
        top_features = fp_df.columns[:100]  # Take first 100 features
        return fp_df[top_features]
    
    def compute_coulomb_matrices(self, smiles_list):
        """Compute Coulomb Matrix representations."""
        print("⚛️ Computing Coulomb Matrices...")
        
        coulomb_features = []
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                print(f"   Processing molecule {i+1}/{len(smiles_list)}")
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    cm = self.coulomb_matrix.get_coulomb_matrix(mol)
                    if cm is not None:
                        coulomb_features.append(cm)
                    else:
                        coulomb_features.append(np.zeros(2500))  # 50x50 flattened
                else:
                    coulomb_features.append(np.zeros(2500))
            except:
                coulomb_features.append(np.zeros(2500))
        
        # Convert to DataFrame
        cm_df = pd.DataFrame(coulomb_features)
        cm_df.columns = [f'CM_{i}' for i in range(len(cm_df.columns))]
        
        # Select top features (reduce dimensionality)
        top_features = cm_df.columns[:100]  # Take first 100 features
        return cm_df[top_features]
    
    def prepare_features(self):
        """Prepare all features for modeling."""
        print("\n🔧 Preparing features...")
        
        if self.cleaned_data is None:
            print("❌ No cleaned data!")
            return False
        
        # Get direct properties
        feature_cols = [col for col in self.cleaned_data.columns if col not in ['smiles', 'gap']]
        X = self.cleaned_data[feature_cols].copy()
        y = self.cleaned_data['gap'].copy()
        
        print(f"📊 Direct features: {len(feature_cols)}")
        print(f"📋 Feature names: {feature_cols}")
        
        # Add RDKit descriptors
        if RDKIT_AVAILABLE:
            rdkit_desc = self.compute_rdkit_descriptors(self.cleaned_data['smiles'])
            if not rdkit_desc.empty:
                rdkit_desc = rdkit_desc.reset_index(drop=True)
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)
                X = pd.concat([X, rdkit_desc], axis=1)
                print(f"🧪 Added {len(rdkit_desc.columns)} RDKit descriptors")
        
        # Add ECFP fingerprints
        if RDKIT_AVAILABLE:
            ecfp_desc = self.compute_ecfp_fingerprints(self.cleaned_data['smiles'])
            if not ecfp_desc.empty:
                ecfp_desc = ecfp_desc.reset_index(drop=True)
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)
                X = pd.concat([X, ecfp_desc], axis=1)
                print(f"🔍 Added {len(ecfp_desc.columns)} ECFP fingerprint features")
        
        # Add Coulomb Matrix features
        if RDKIT_AVAILABLE:
            cm_desc = self.compute_coulomb_matrices(self.cleaned_data['smiles'])
            if not cm_desc.empty:
                cm_desc = cm_desc.reset_index(drop=True)
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)
                X = pd.concat([X, cm_desc], axis=1)
                print(f"⚛️ Added {len(cm_desc.columns)} Coulomb Matrix features")
        
        # Handle missing values
        missing_count = X.isnull().sum().sum()
        print(f"❓ Missing values: {missing_count}")
        X = X.fillna(0)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Features prepared!")
        print(f"📊 Total features: {len(self.feature_names)}")
        print(f"📊 Training set: {self.X_train.shape}")
        print(f"📊 Test set: {self.X_test.shape}")
        
        return True
    
    def train_baseline_models(self):
        """Train baseline models as required."""
        print("\n🤖 Training baseline models...")
        
        # 1. Random Forest Regressor
        print("🌲 Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train_scaled, self.y_train)
        self.models['RandomForest'] = rf_model
        
        # 2. MLP Regressor (Keras equivalent using sklearn)
        print("🧠 Training MLP...")
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        mlp_model.fit(self.X_train_scaled, self.y_train)
        self.models['MLP'] = mlp_model
        
        print("✅ Baseline models trained!")
    
    def train_gnn_model(self):
        """Train Graph Neural Network model."""
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch not available, skipping GNN")
            return
        
        print("\n🌲 Training Graph Neural Network...")
        
        # Import GNN components
        from QM9_GNN_SIMPLE import SimpleQM9GNNPipeline
        
        # Create and train GNN
        gnn_pipeline = SimpleQM9GNNPipeline('qm9_cleaned_complete.csv')
        gnn_pipeline.load_data()
        gnn_pipeline.clean_data()
        
        # Get GNN predictions
        X_gnn = gnn_pipeline.cleaned_data[['smiles']]
        y_gnn = gnn_pipeline.cleaned_data['gap']
        
        X_gnn_train, X_gnn_test, y_gnn_train, y_gnn_test = train_test_split(
            X_gnn, y_gnn, test_size=0.2, random_state=42
        )
        
        # Create graph datasets
        train_graphs, _ = gnn_pipeline.create_graph_dataset(
            X_gnn_train['smiles'].values, y_gnn_train.values
        )
        test_graphs, _ = gnn_pipeline.create_graph_dataset(
            X_gnn_test['smiles'].values, y_gnn_test.values
        )
        
        if len(train_graphs) > 0 and len(test_graphs) > 0:
            # Train GNN
            val_preds, val_targets = gnn_pipeline.train_gnn_model(train_graphs, test_graphs, epochs=20)
            
            # Store GNN results
            self.gnn_results = {
                'predictions': val_preds,
                'targets': val_targets,
                'model': gnn_pipeline.model
            }
            print("✅ GNN model trained!")
        else:
            print("❌ Could not create valid graphs for GNN")
    
    def evaluate_models(self):
        """Evaluate all models and create plots."""
        print("\n📊 Evaluating models...")
        
        results = {}
        
        # Evaluate baseline models
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test_scaled)
            
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            
            results[name] = {
                'predictions': y_pred,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"📈 {name} Performance:")
            print(f"   MAE: {mae:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   R²: {r2:.4f}")
        
        # Add GNN results if available
        if hasattr(self, 'gnn_results'):
            gnn_preds = self.gnn_results['predictions']
            gnn_targets = self.gnn_results['targets']
            
            mae = mean_absolute_error(gnn_targets, gnn_preds)
            rmse = np.sqrt(mean_squared_error(gnn_targets, gnn_preds))
            r2 = r2_score(gnn_targets, gnn_preds)
            
            results['GNN'] = {
                'predictions': gnn_preds,
                'targets': gnn_targets,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"📈 GNN Performance:")
            print(f"   MAE: {mae:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   R²: {r2:.4f}")
        
        self.results = results
        return results
    
    def create_plots(self):
        """Create all required plots."""
        print("\n📊 Creating plots...")
        
        # 1. Actual vs Predicted plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(self.results.keys())
        for i, (name, result) in enumerate(self.results.items()):
            row = i // 2
            col = i % 2
            
            if name == 'GNN':
                y_true = result['targets']
                y_pred = result['predictions']
            else:
                y_true = self.y_test
                y_pred = result['predictions']
            
            axes[row, col].scatter(y_true, y_pred, alpha=0.6)
            axes[row, col].plot([min(y_true), max(y_true)], 
                               [min(y_true), max(y_true)], 'r--', lw=2)
            axes[row, col].set_xlabel('Actual HOMO-LUMO Gap')
            axes[row, col].set_ylabel('Predicted HOMO-LUMO Gap')
            axes[row, col].set_title(f'{name}: Actual vs Predicted')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qm9_complete_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature importance plot (for Random Forest)
        if 'RandomForest' in self.models:
            rf_model = self.models['RandomForest']
            feature_importance = rf_model.feature_importances_
            
            # Get top 20 features
            top_indices = np.argsort(feature_importance)[-20:]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Most Important Features (Random Forest)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('qm9_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("✅ Plots saved!")
    
    def save_models(self):
        """Save all trained models."""
        print("\n💾 Saving models...")
        
        # Save baseline models
        for name, model in self.models.items():
            joblib.dump(model, f'qm9_{name.lower()}_model.joblib')
            print(f"✅ {name} model saved")
        
        # Save GNN model if available
        if hasattr(self, 'gnn_results'):
            torch.save(self.gnn_results['model'].state_dict(), 'qm9_gnn_model.pth')
            print("✅ GNN model saved")
        
        # Save scaler
        joblib.dump(self.scaler, 'qm9_scaler.joblib')
        print("✅ Scaler saved")
        
        # Save feature names
        joblib.dump(self.feature_names, 'qm9_feature_names.joblib')
        print("✅ Feature names saved")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline."""
        print("🚀 Starting Complete QM9 Pipeline - Task Group 3")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Clean data
        if not self.clean_data():
            return False
        
        # Prepare features
        if not self.prepare_features():
            return False
        
        # Train models
        self.train_baseline_models()
        self.train_gnn_model()
        
        # Evaluate models
        results = self.evaluate_models()
        
        # Create plots
        self.create_plots()
        
        # Save models
        self.save_models()
        
        print("\n🎉 Complete Pipeline finished!")
        print("=" * 60)
        
        return results

def main():
    """Main function."""
    print("🔬 QM9 Complete AI Model Pipeline - Task Group 3")
    print("=" * 50)
    
    pipeline = CompleteQM9Pipeline()
    results = pipeline.run_complete_pipeline()
    
    if results:
        print(f"\n📋 Final Results Summary:")
        for name, result in results.items():
            print(f"   {name}: R² = {result['r2']:.4f}, MAE = {result['mae']:.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"\n🏆 Best Model: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")

if __name__ == "__main__":
    main() 