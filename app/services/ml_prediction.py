"""
Machine Learning prediction models for startup success analysis.
"""
import logging
import asyncio
import pickle
import joblib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from app.services.feature_engineering import AdvancedFeatureEngine, StartupFeatures

logger = logging.getLogger(__name__)


class MLPredictionEngine:
    """Advanced ML prediction models for startup success analysis."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_dir = Path("ml_models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.feature_engine = AdvancedFeatureEngine()
        
        # Define success thresholds
        self.success_thresholds = {
            'funding_success': 1000000,  # $1M+ funding
            'growth_success': 50.0,      # 50%+ growth rate
            'exit_success': 50000000,    # $50M+ exit value
            'sustainability': 24         # 24+ months survival
        }
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model_class': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [8, 10, 12],
                    'min_samples_split': [3, 5, 7]
                }
            },
            'xgboost': {
                'model_class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.05, 0.1, 0.15]
                }
            },
            'gradient_boosting': {
                'model_class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'random_state': 42
                },
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.05, 0.1, 0.15]
                }
            }
        }
    
    async def train_success_prediction_models(
        self, 
        training_data: List[Dict[str, Any]],
        target_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Train ML models for startup success prediction."""
        if target_metrics is None:
            target_metrics = ['overall_success', 'funding_success', 'growth_success']
        
        logger.info(f"Training ML models on {len(training_data)} startups")
        
        # Prepare training data
        features_data = []
        targets_data = {metric: [] for metric in target_metrics}
        
        for startup_data in training_data:
            # Extract features
            features = await self.feature_engine.extract_startup_features(startup_data)
            feature_dict = self.feature_engine.features_to_dict(features)
            features_data.append(feature_dict)
            
            # Extract target labels
            for metric in target_metrics:
                success_label = self._determine_success_label(startup_data, metric)
                targets_data[metric].append(success_label)
        
        if not features_data:
            raise ValueError("No valid training data available")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_data)
        
        # Train models for each target metric
        training_results = {}
        
        for metric in target_metrics:
            logger.info(f"Training models for {metric}")
            
            targets = np.array(targets_data[metric])
            
            # Skip if not enough positive examples
            if np.sum(targets) < 5:
                logger.warning(f"Not enough positive examples for {metric}, skipping")
                continue
            
            metric_results = await self._train_models_for_metric(
                features_df, targets, metric
            )
            training_results[metric] = metric_results
        
        return training_results
    
    def _determine_success_label(self, startup_data: Dict[str, Any], metric: str) -> int:
        """Determine success label based on startup data and metric."""
        if metric == 'overall_success':
            # Composite success based on multiple factors
            score = 0
            
            # Funding success
            if self._check_funding_success(startup_data):
                score += 1
            
            # Growth indicators
            if self._check_growth_success(startup_data):
                score += 1
            
            # Market traction
            if self._check_traction_success(startup_data):
                score += 1
            
            # Team quality
            if self._check_team_quality(startup_data):
                score += 1
            
            return 1 if score >= 2 else 0
        
        elif metric == 'funding_success':
            return 1 if self._check_funding_success(startup_data) else 0
        
        elif metric == 'growth_success':
            return 1 if self._check_growth_success(startup_data) else 0
        
        elif metric == 'exit_success':
            return 1 if self._check_exit_success(startup_data) else 0
        
        else:
            return 0
    
    def _check_funding_success(self, startup_data: Dict[str, Any]) -> bool:
        """Check if startup achieved funding success."""
        # Look for funding indicators in description or analysis
        text_content = ' '.join([
            startup_data.get('description', ''),
            str(startup_data.get('analysis', {}))
        ]).lower()
        
        funding_indicators = [
            'series a', 'series b', 'series c', 'funding round',
            'million raised', 'investment', 'venture capital',
            'seed funding', 'angel investment'
        ]
        
        return any(indicator in text_content for indicator in funding_indicators)
    
    def _check_growth_success(self, startup_data: Dict[str, Any]) -> bool:
        """Check if startup shows strong growth."""
        analysis = startup_data.get('analysis', {})
        traction = analysis.get('traction_analysis', {})
        
        growth_indicators = [
            'rapid growth', 'exponential growth', 'viral adoption',
            'scaling quickly', 'fast growing', 'market leader'
        ]
        
        traction_text = traction.get('metrics', '').lower()
        
        return any(indicator in traction_text for indicator in growth_indicators)
    
    def _check_traction_success(self, startup_data: Dict[str, Any]) -> bool:
        """Check if startup has strong traction."""
        website_data = startup_data.get('website_data', {})
        customer_logos = website_data.get('customer_logos', [])
        
        # Strong traction indicators
        if len(customer_logos) > 5:
            return True
        
        # Check for pricing (indicates paying customers)
        pricing_info = website_data.get('pricing_info', {})
        if pricing_info.get('plans') and not pricing_info.get('free_tier'):
            return True
        
        return False
    
    def _check_team_quality(self, startup_data: Dict[str, Any]) -> bool:
        """Check if startup has high-quality team."""
        founders = startup_data.get('founders', [])
        
        quality_indicators = 0
        
        for founder in founders:
            background = founder.get('background', '').lower()
            
            # Previous startup experience
            if any(word in background for word in ['founder', 'startup', 'entrepreneur']):
                quality_indicators += 1
            
            # Big tech experience
            if any(company in background for company in ['google', 'facebook', 'amazon', 'microsoft']):
                quality_indicators += 1
            
            # Technical background
            if any(word in background for word in ['engineer', 'cto', 'technical', 'developer']):
                quality_indicators += 1
        
        return quality_indicators >= 2
    
    def _check_exit_success(self, startup_data: Dict[str, Any]) -> bool:
        """Check if startup achieved successful exit."""
        text_content = ' '.join([
            startup_data.get('description', ''),
            str(startup_data.get('analysis', {}))
        ]).lower()
        
        exit_indicators = [
            'acquired', 'acquisition', 'ipo', 'public offering',
            'sold to', 'exit', 'billion dollar', 'unicorn'
        ]
        
        return any(indicator in text_content for indicator in exit_indicators)
    
    async def _train_models_for_metric(
        self, 
        features_df: pd.DataFrame, 
        targets: np.ndarray, 
        metric: str
    ) -> Dict[str, Any]:
        """Train all models for a specific success metric."""
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets, test_size=0.2, random_state=42, stratify=targets
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[metric] = scaler
        
        # Train each model type
        model_results = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name} for {metric}")
            
            try:
                # Train base model
                model = config['model_class'](**config['params'])
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                # Predictions for detailed metrics
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate AUC if available
                auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(features_df.columns, model.feature_importances_))
                    # Sort by importance
                    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
                else:
                    importance = {}
                
                # Store model and results
                self.models[f"{metric}_{model_name}"] = model
                
                model_results[model_name] = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'auc_score': auc_score,
                    'feature_importance': importance,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                # Hyperparameter tuning for best model
                if model_name in ['xgboost', 'random_forest']:  # Only tune top models
                    tuned_model = await self._tune_hyperparameters(
                        config, X_train_scaled, y_train, model_name
                    )
                    if tuned_model:
                        self.models[f"{metric}_{model_name}_tuned"] = tuned_model
                        
                        # Evaluate tuned model
                        tuned_test_score = tuned_model.score(X_test_scaled, y_test)
                        model_results[model_name]['tuned_test_accuracy'] = tuned_test_score
                
            except Exception as e:
                logger.error(f"Failed to train {model_name} for {metric}: {e}")
                model_results[model_name] = {'error': str(e)}
        
        # Save models and scalers
        await self._save_models_and_scalers(metric)
        
        return model_results
    
    async def _tune_hyperparameters(
        self, 
        config: Dict[str, Any], 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        model_name: str
    ) -> Optional[Any]:
        """Tune hyperparameters using GridSearchCV."""
        try:
            logger.info(f"Tuning hyperparameters for {model_name}")
            
            model = config['model_class']()
            
            # Use smaller parameter grid for speed
            param_grid = config['param_grid']
            
            # Run grid search in thread pool
            loop = asyncio.get_event_loop()
            grid_search = await loop.run_in_executor(
                None,
                lambda: GridSearchCV(
                    model, 
                    param_grid, 
                    cv=3, 
                    scoring='roc_auc', 
                    n_jobs=-1
                ).fit(X_train, y_train)
            )
            
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed for {model_name}: {e}")
            return None
    
    async def _save_models_and_scalers(self, metric: str):
        """Save trained models and scalers to disk."""
        try:
            # Save models
            for model_key, model in self.models.items():
                if metric in model_key:
                    model_file = self.model_dir / f"{model_key}.pkl"
                    joblib.dump(model, model_file)
            
            # Save scaler
            if metric in self.scalers:
                scaler_file = self.model_dir / f"{metric}_scaler.pkl"
                joblib.dump(self.scalers[metric], scaler_file)
            
            logger.info(f"Saved models and scaler for {metric}")
            
        except Exception as e:
            logger.error(f"Failed to save models for {metric}: {e}")
    
    async def load_models(self, metric: str) -> bool:
        """Load trained models and scalers from disk."""
        try:
            # Load scaler
            scaler_file = self.model_dir / f"{metric}_scaler.pkl"
            if scaler_file.exists():
                self.scalers[metric] = joblib.load(scaler_file)
            
            # Load models
            for model_file in self.model_dir.glob(f"{metric}_*.pkl"):
                if "scaler" not in model_file.name:
                    model_key = model_file.stem
                    self.models[model_key] = joblib.load(model_file)
            
            logger.info(f"Loaded models for {metric}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models for {metric}: {e}")
            return False
    
    async def predict_startup_success(
        self, 
        startup_data: Dict[str, Any],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Predict startup success across multiple metrics."""
        if metrics is None:
            metrics = ['overall_success', 'funding_success', 'growth_success']
        
        # Extract features
        features = await self.feature_engine.extract_startup_features(startup_data)
        feature_dict = self.feature_engine.features_to_dict(features)
        feature_array = np.array([list(feature_dict.values())])
        
        predictions = {}
        
        for metric in metrics:
            metric_predictions = {}
            
            # Try to load models if not already loaded
            if not any(metric in key for key in self.models.keys()):
                await self.load_models(metric)
            
            # Scale features
            if metric in self.scalers:
                scaler = self.scalers[metric]
                feature_array_scaled = scaler.transform(feature_array)
            else:
                feature_array_scaled = feature_array
                logger.warning(f"No scaler found for {metric}, using unscaled features")
            
            # Make predictions with all available models
            for model_key, model in self.models.items():
                if metric in model_key:
                    try:
                        # Probability prediction
                        if hasattr(model, 'predict_proba'):
                            prob = model.predict_proba(feature_array_scaled)[0][1]
                        else:
                            prob = model.decision_function(feature_array_scaled)[0]
                            # Convert to probability-like score
                            prob = 1 / (1 + np.exp(-prob))
                        
                        # Binary prediction
                        prediction = model.predict(feature_array_scaled)[0]
                        
                        model_name = model_key.replace(f"{metric}_", "")
                        metric_predictions[model_name] = {
                            'probability': float(prob),
                            'prediction': int(prediction),
                            'confidence': float(abs(prob - 0.5) * 2)  # Distance from neutral
                        }
                        
                    except Exception as e:
                        logger.error(f"Prediction failed for model {model_key}: {e}")
            
            # Ensemble prediction (average of all models)
            if metric_predictions:
                avg_probability = np.mean([pred['probability'] for pred in metric_predictions.values()])
                avg_prediction = 1 if avg_probability > 0.5 else 0
                avg_confidence = np.mean([pred['confidence'] for pred in metric_predictions.values()])
                
                metric_predictions['ensemble'] = {
                    'probability': float(avg_probability),
                    'prediction': int(avg_prediction),
                    'confidence': float(avg_confidence)
                }
            
            predictions[metric] = metric_predictions
        
        # Calculate overall success score
        overall_score = self._calculate_overall_success_score(predictions)
        
        return {
            'predictions': predictions,
            'overall_success_score': overall_score,
            'feature_values': feature_dict,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_overall_success_score(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall success score from individual predictions."""
        scores = []
        weights = {
            'overall_success': 0.4,
            'funding_success': 0.3,
            'growth_success': 0.2,
            'exit_success': 0.1
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for metric, weight in weights.items():
            if metric in predictions and 'ensemble' in predictions[metric]:
                prob = predictions[metric]['ensemble']['probability']
                weighted_score += prob * weight * 100  # Convert to 0-100 scale
                total_weight += weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 50.0  # Neutral score if no predictions available
    
    async def get_prediction_explanation(
        self, 
        startup_data: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate explanation for predictions."""
        # Extract features for analysis
        features = await self.feature_engine.extract_startup_features(startup_data)
        feature_dict = self.feature_engine.features_to_dict(features)
        
        # Get feature importance from models
        explanations = {}
        
        for metric, metric_predictions in predictions['predictions'].items():
            if not metric_predictions:
                continue
                
            # Get feature importance from best performing model
            best_model_key = f"{metric}_xgboost"  # Prefer XGBoost if available
            if best_model_key not in self.models:
                # Fall back to any available model
                available_models = [key for key in self.models.keys() if metric in key]
                if available_models:
                    best_model_key = available_models[0]
                else:
                    continue
            
            model = self.models[best_model_key]
            
            if hasattr(model, 'feature_importances_'):
                # Get top influential features
                importance = model.feature_importances_
                feature_names = list(feature_dict.keys())
                
                # Sort features by importance
                feature_importance = list(zip(feature_names, importance, 
                                            [feature_dict[name] for name in feature_names]))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Generate explanations for top features
                top_features = feature_importance[:5]
                feature_explanations = []
                
                for feature_name, importance_score, feature_value in top_features:
                    explanation = self._explain_feature_impact(
                        feature_name, feature_value, importance_score
                    )
                    feature_explanations.append({
                        'feature': feature_name,
                        'value': feature_value,
                        'importance': float(importance_score),
                        'explanation': explanation
                    })
                
                explanations[metric] = {
                    'prediction_summary': self._summarize_prediction(metric_predictions.get('ensemble', {})),
                    'key_factors': feature_explanations,
                    'overall_assessment': self._generate_overall_assessment(metric, metric_predictions)
                }
        
        return explanations
    
    def _explain_feature_impact(self, feature_name: str, feature_value: float, importance: float) -> str:
        """Explain the impact of a specific feature."""
        # Map feature names to explanations
        feature_explanations = {
            'team_size': f"Team has {int(feature_value)} members",
            'founder_experience_score': f"Founder experience rated {feature_value:.1f}/100",
            'technical_founder_present': "Technical founder present" if feature_value > 0 else "No technical founder",
            'education_prestige_score': f"Education prestige score: {feature_value:.1f}/100",
            'market_size_score': f"Market size potential: {feature_value:.1f}/100",
            'product_complexity_score': f"Product complexity: {feature_value:.1f}/100",
            'github_activity_score': f"GitHub activity level: {feature_value:.1f}/100",
            'customer_validation_score': f"Customer validation: {feature_value:.1f}/100",
            'revenue_model_strength': f"Revenue model strength: {feature_value:.1f}/100"
        }
        
        base_explanation = feature_explanations.get(
            feature_name, 
            f"{feature_name.replace('_', ' ').title()}: {feature_value:.1f}"
        )
        
        # Add impact assessment
        if importance > 0.1:
            impact = "High impact factor"
        elif importance > 0.05:
            impact = "Moderate impact factor"
        else:
            impact = "Minor impact factor"
        
        return f"{base_explanation} ({impact})"
    
    def _summarize_prediction(self, ensemble_prediction: Dict[str, Any]) -> str:
        """Summarize ensemble prediction."""
        if not ensemble_prediction:
            return "No prediction available"
        
        probability = ensemble_prediction.get('probability', 0.5)
        confidence = ensemble_prediction.get('confidence', 0.0)
        
        if probability > 0.7:
            likelihood = "High"
        elif probability > 0.5:
            likelihood = "Moderate"
        else:
            likelihood = "Low"
        
        if confidence > 0.7:
            conf_level = "high"
        elif confidence > 0.4:
            conf_level = "moderate"
        else:
            conf_level = "low"
        
        return f"{likelihood} likelihood of success ({probability:.1%}) with {conf_level} confidence"
    
    def _generate_overall_assessment(self, metric: str, predictions: Dict[str, Any]) -> str:
        """Generate overall assessment for the metric."""
        if not predictions:
            return "Insufficient data for assessment"
        
        ensemble = predictions.get('ensemble', {})
        if not ensemble:
            return "No ensemble prediction available"
        
        probability = ensemble.get('probability', 0.5)
        
        if metric == 'overall_success':
            if probability > 0.7:
                return "Strong potential for success across multiple dimensions"
            elif probability > 0.5:
                return "Moderate success potential with room for improvement"
            else:
                return "Significant challenges identified requiring strategic focus"
        
        elif metric == 'funding_success':
            if probability > 0.7:
                return "High likelihood of securing funding based on current metrics"
            elif probability > 0.5:
                return "Reasonable funding prospects with some areas needing strengthening"
            else:
                return "Funding challenges likely without significant improvements"
        
        elif metric == 'growth_success':
            if probability > 0.7:
                return "Strong growth indicators suggest rapid scaling potential"
            elif probability > 0.5:
                return "Moderate growth potential with focused execution"
            else:
                return "Growth challenges require strategic pivoting or optimization"
        
        return "Assessment pending additional data"
    
    async def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance across all metrics."""
        summary = {
            'available_models': list(self.models.keys()),
            'metrics_covered': list(set(key.split('_')[0] for key in self.models.keys())),
            'model_types': list(set('_'.join(key.split('_')[1:]) for key in self.models.keys())),
            'total_models': len(self.models),
            'feature_importance_summary': {}
        }
        
        # Aggregate feature importance across models
        all_importance = {}
        
        for model_key, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # This would need actual feature names - simplified for now
                importance_scores = model.feature_importances_
                for i, score in enumerate(importance_scores):
                    feature_name = f"feature_{i}"  # Would be actual feature names
                    all_importance[feature_name] = all_importance.get(feature_name, []) + [score]
        
        # Calculate average importance
        for feature, scores in all_importance.items():
            summary['feature_importance_summary'][feature] = {
                'average_importance': np.mean(scores),
                'std_importance': np.std(scores),
                'models_count': len(scores)
            }
        
        return summary