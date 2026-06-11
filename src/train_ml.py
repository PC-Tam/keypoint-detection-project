import yaml
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def get_models(config):
    """
    Initialize and return the dictionary of ML models based on config.
    """
    rs = config['models']['random_state']
    
    models = {
        "SVM": SVC(
            kernel=config['models']['svm']['kernel'], 
            C=config['models']['svm']['C'], 
            probability=True, 
            random_state=rs
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=config['models']['random_forest']['n_estimators'], 
            random_state=rs
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=config['models']['knn']['n_neighbors']
        ),
        "GNB": GaussianNB()
    }
    return models

def train_models(X_train, y_train, feature_name):
    """
    Train all models on the given feature set and save them.
    Args:
        X_train: Training features.
        y_train: Training labels.
        feature_name: Name of the feature (e.g., 'harris', 'fast', 'orb_bovw').
    Returns:
        trained_models: Dictionary of trained models.
    """
    config = load_config()
    models = get_models(config)
    
    save_dir = Path("outputs/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name} on {feature_name} features...")
        model.fit(X_train, y_train)
        
        # Save model
        model_path = save_dir / f"{model_name}_{feature_name}.joblib"
        joblib.dump(model, model_path)
        
        trained_models[model_name] = model
        
    return trained_models
