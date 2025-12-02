import copy
import os
import argparse
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import spearmanr
import joblib
import torch
from utils.utils import parse_config, set_random_seed


def load_data(data_dir, backbone_model='ProtAlbert', target_col='log_fitness'):
    """Load embedding data and target values"""
    df_data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
    embd_file = os.path.join(data_dir, f'{backbone_model}/{backbone_model}.npy')

    if os.path.exists(embd_file):
        prot_embd = np.load(embd_file)
    elif backbone_model.startswith('esm'):
        layer = 36 if 't36' in backbone_model else 33 # only support esm2_t36 and esm2_t33 for now
        prot_embd = []
        # for id in range(len(df_data['seq'])):
            # emb = torch.load(os.path.join(data_dir, backbone_model, f'seq_{id}.pt'))['mean_representations'][layer]
        for id in df_data['ID']:
            emb = torch.load(os.path.join(data_dir, backbone_model, f'{id}.pt'))['mean_representations'][layer]
            prot_embd.append(emb)
        prot_embd = torch.stack(prot_embd, dim=0).numpy()
        np.save(os.path.join(data_dir, backbone_model, f'{backbone_model}.npy'), prot_embd)
    else:
        raise FileNotFoundError(f"Embedding file {embd_file} not found")

    Xs = prot_embd
    Ys = df_data[target_col].values
    
    return Xs, Ys


def apply_pca(Xs_train, Xs_test=None, n_components=60, output_dir=None, save_model=True):
    """Apply PCA transformation to training and test data"""
    from sklearn.decomposition import PCA
    
    pca_model = PCA(n_components=n_components)
    Xs_train_pca = pca_model.fit_transform(Xs_train)
    
    if save_model and output_dir:
        joblib.dump(pca_model, os.path.join(output_dir, 'pca_model.joblib'))
    
    if Xs_test is not None:
        Xs_test_pca = pca_model.transform(Xs_test)
        return Xs_train_pca, Xs_test_pca, pca_model
    else:
        return Xs_train_pca, None, pca_model


def create_reg_model(model_type):
    """Create regression model based on type"""
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()

    elif model_type == 'knn':
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor()

    elif model_type == 'svm':
        from sklearn.svm import SVR
        model = SVR()

    elif model_type == 'gbdt':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor()

    elif model_type == 'sgd':
        from sklearn.linear_model import SGDRegressor
        model = SGDRegressor()

    elif model_type == 'guass_nb':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()

    elif model_type == 'mlp':
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor()
    
    elif model_type == 'xgboost':
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()
        except ImportError:
            raise ImportError("XGBoost not installed. Please install with: pip install xgboost")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model


def perform_cross_validation(Xs, Ys, args):
    """Perform 5-fold cross validation"""
    
    # Setup 5-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = []
    fold_results = []
    
    print(f"Performing 5-fold cross validation...")
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(Xs)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        
        # Split data for this fold
        Xs_train_fold = Xs[train_idx]
        Xs_test_fold = Xs[test_idx]
        ys_train_fold = Ys[train_idx]
        ys_test_fold = Ys[test_idx]
        
        # Create fold-specific output directory
        fold_output_dir = os.path.join(args.output_dir, f'fold_{fold + 1}')
        Path(fold_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Apply PCA if specified
        if args.n_components is not None:
            Xs_train_fold, Xs_test_fold, pca_model = apply_pca(
                Xs_train_fold, Xs_test_fold, 
                n_components=args.n_components, 
                output_dir=fold_output_dir, 
                save_model=True
            )
        
        # Create and train model for this fold
        head_model = create_reg_model(args.head_model)
        head_model.fit(Xs_train_fold, ys_train_fold)
        
        # Make predictions
        ys_test_pred_fold = head_model.predict(Xs_test_fold)
        
        # Calculate Spearman correlation for this fold
        spearman_r_fold = spearmanr(ys_test_fold, ys_test_pred_fold)[0]
        cv_scores.append(spearman_r_fold)
        
        print(f"Spearman correlation: {spearman_r_fold:.3f}")
        
        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'spearman_r': spearman_r_fold,
            'train_size': len(ys_train_fold),
            'test_size': len(ys_test_fold)
        })
    
    return cv_scores, fold_results


def perform_single_split(Xs, Ys, args):
    """Perform single train-test split (original behavior)"""
    
    # Random train-test split
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        Xs, Ys, test_size=args.test_size, random_state=args.seed)
    
    # Apply PCA if specified
    if args.n_components is not None:
        Xs_train, _, pca_model = apply_pca(
            Xs_train, 
            n_components=args.n_components, 
            output_dir=args.output_dir, 
            save_model=True
        )
        Xs_test = pca_model.transform(Xs_test)
    
    # Create and train model
    head_model = create_reg_model(args.head_model)
    head_model.fit(Xs_train, ys_train)
    
    # Save model
    joblib.dump(head_model, os.path.join(args.output_dir, f'head_model_{args.head_model}.joblib'))
    
    # Make predictions
    ys_test_pred = head_model.predict(Xs_test)
    ys_train_pred = head_model.predict(Xs_train)
    
    # Calculate Spearman correlation
    spearman_r = spearmanr(ys_test, ys_test_pred)[0]
    
    # Save predictions
    np.save(os.path.join(args.output_dir, 'ys_test_pred.npy'), ys_test_pred)
    np.save(os.path.join(args.output_dir, 'ys_test.npy'), ys_test)
    np.save(os.path.join(args.output_dir, 'ys_train_pred.npy'), ys_train_pred)
    np.save(os.path.join(args.output_dir, 'ys_train.npy'), ys_train)
    
    return spearman_r


def get_args():
    parser = argparse.ArgumentParser(description='Protein fitness modeling via protein language models with 5-fold cross validation support')
    
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('-s', '--seed', type=int, default=27, help='Random seed, default=27')
    parser.add_argument('-b', '--backbone_model', type=str, default='ProtT5-XL', help="Backbone model type in 'ProtT5-XL', 'ProtAlbert', 'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D', 'esm1v_t33_650M_UR90S_1', 'esm1v_t33_650M_UR90S_2', 'esm1v_t33_650M_UR90S_3', 'esm1v_t33_650M_UR90S_4', and 'esm1v_t33_650M_UR90S_5', default=ProtT5-XL")
    parser.add_argument('-m', '--head_model', type=str, default='svm', help='Head model type, default=svm')
    parser.add_argument('-n', '--n_components', type=int, default=60, help='Number of PCA components, default=60')
    parser.add_argument('-cv', '--cross_val', action='store_true', help='Perform 5-fold cross validation, default=False')
    parser.add_argument('--target_col', type=str, default='log_fitness', help='Target column name, default=log_fitness')
    parser.add_argument('--test_size', type=float, default=0.2, help='Train-test split ratio, default 0.2 for testing')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Load data
    print(f"Loading data from {args.input_dir}...")
    Xs, Ys = load_data(args.input_dir, args.backbone_model, args.target_col)
    print(f"Loaded {len(Xs)} samples with {Xs.shape[1]} features")
    print(f"Backbone model: {args.backbone_model}")
    print(f"Head model: {args.head_model}")
    print(f"PCA components: {args.n_components}")
    print(f"Random seed: {args.seed}")
    
    if args.cross_val:
        # Perform 5-fold cross validation
        cv_scores, fold_results = perform_cross_validation(Xs, Ys, args)
        
        # Calculate and print cross validation statistics
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        print(f"\n=== 5-Fold Cross Validation Results ===")
        print(f"Mean CV score: {mean_cv_score:.3f} ± {std_cv_score:.3f}")
        print(f"Range: {np.max(cv_scores):.3f} - {np.min(cv_scores):.3f}")
        
        # Save summary as CSV
        df_results = pd.DataFrame(fold_results)
        df_results.to_csv(os.path.join(args.output_dir, f'{args.backbone_model}_{args.head_model}_cv_results.csv'), index=False)
        
        
    else:
        # Perform single split (original behavior)
        print(f"Performing single train-test split...")
        spearman_r = perform_single_split(Xs, Ys, args)
        print(f"Spearman correlation: {spearman_r:.3f}")
        
        # Print final result in the original format for compatibility
        print(f"{args.seed}, {args.backbone_model}, {args.head_model}, {spearman_r:.3f}")


if __name__ == "__main__":
    main()
