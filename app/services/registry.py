"""Model registry and versioning services."""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import mlflow
from mlflow.tracking import MlflowClient

from ..config import MLRUNS_DIR, BASE_DIR

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Service for managing model versions and deployments."""
    
    def __init__(self):
        self.registry_dir = BASE_DIR / "models"
        self.registry_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.registry_dir / "registry.json"
        self.client = MlflowClient()
        
        # Initialize registry metadata if it doesn't exist
        if not self.metadata_file.exists():
            self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the model registry metadata file."""
        initial_data = {
            "models": {},
            "deployments": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
        
        logger.info("Model registry initialized")
    
    async def register_model(
        self,
        model_name: str,
        run_id: str,
        model_type: str,
        location: str,
        metrics: Dict[str, float],
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register a new model version.
        
        Args:
            model_name: Name of the model
            run_id: MLflow run ID
            model_type: Type of the model
            location: Location the model was trained for
            metrics: Model performance metrics
            description: Optional description
            tags: Optional tags
            
        Returns:
            Model version ID
        """
        try:
            # Generate version ID
            version_id = f"{model_name}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Load current registry
            registry_data = self._load_registry()
            
            # Create model entry if it doesn't exist
            if model_name not in registry_data["models"]:
                registry_data["models"][model_name] = {
                    "created_at": datetime.now().isoformat(),
                    "versions": {},
                    "active_version": None,
                    "total_versions": 0
                }
            
            # Add new version
            version_data = {
                "version_id": version_id,
                "run_id": run_id,
                "model_type": model_type,
                "location": location,
                "metrics": metrics,
                "description": description or "",
                "tags": tags or {},
                "created_at": datetime.now().isoformat(),
                "status": "registered",
                "deployment_status": "none"
            }
            
            registry_data["models"][model_name]["versions"][version_id] = version_data
            registry_data["models"][model_name]["total_versions"] += 1
            registry_data["last_updated"] = datetime.now().isoformat()
            
            # Set as active version if it's the first or if it's better than current
            current_active = registry_data["models"][model_name]["active_version"]
            if current_active is None or self._is_better_model(version_data, registry_data["models"][model_name]["versions"].get(current_active)):
                registry_data["models"][model_name]["active_version"] = version_id
                version_data["status"] = "active"
            
            # Save registry
            self._save_registry(registry_data)
            
            # Register with MLflow if available
            try:
                mlflow.register_model(
                    f"runs:/{run_id}/model",
                    model_name,
                    description=description
                )
            except Exception as e:
                logger.warning(f"Failed to register with MLflow: {e}")
            
            logger.info(f"Model {model_name} version {version_id} registered successfully")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    async def deploy_model(
        self,
        model_name: str,
        version_id: str,
        deployment_target: str = "production",
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Deploy a model version.
        
        Args:
            model_name: Name of the model
            version_id: Version ID to deploy
            deployment_target: Target environment (e.g., 'production', 'staging')
            deployment_config: Deployment configuration
            
        Returns:
            Deployment information
        """
        try:
            registry_data = self._load_registry()
            
            if model_name not in registry_data["models"]:
                raise ValueError(f"Model {model_name} not found")
            
            if version_id not in registry_data["models"][model_name]["versions"]:
                raise ValueError(f"Version {version_id} not found for model {model_name}")
            
            # Create deployment record
            deployment_id = f"deploy_{deployment_target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            deployment_data = {
                "deployment_id": deployment_id,
                "model_name": model_name,
                "version_id": version_id,
                "target": deployment_target,
                "config": deployment_config or {},
                "deployed_at": datetime.now().isoformat(),
                "status": "deployed",
                "endpoint": f"/models/{model_name}/{deployment_target}"
            }
            
            # Update deployment status
            registry_data["models"][model_name]["versions"][version_id]["deployment_status"] = "deployed"
            registry_data["deployments"][deployment_id] = deployment_data
            registry_data["last_updated"] = datetime.now().isoformat()
            
            # Save registry
            self._save_registry(registry_data)
            
            logger.info(f"Model {model_name} version {version_id} deployed to {deployment_target}")
            return deployment_data
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model and its versions.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information
        """
        try:
            registry_data = self._load_registry()
            
            if model_name not in registry_data["models"]:
                raise ValueError(f"Model {model_name} not found")
            
            model_data = registry_data["models"][model_name]
            
            # Add deployment information
            deployments = []
            for deployment_id, deployment_data in registry_data["deployments"].items():
                if deployment_data["model_name"] == model_name:
                    deployments.append(deployment_data)
            
            return {
                "model_name": model_name,
                "created_at": model_data["created_at"],
                "total_versions": model_data["total_versions"],
                "active_version": model_data["active_version"],
                "versions": model_data["versions"],
                "deployments": deployments
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models.
        
        Returns:
            List of model summaries
        """
        try:
            registry_data = self._load_registry()
            
            models = []
            for model_name, model_data in registry_data["models"].items():
                active_version_data = None
                if model_data["active_version"]:
                    active_version_data = model_data["versions"].get(model_data["active_version"])
                
                model_summary = {
                    "model_name": model_name,
                    "created_at": model_data["created_at"],
                    "total_versions": model_data["total_versions"],
                    "active_version": model_data["active_version"],
                    "active_version_metrics": active_version_data["metrics"] if active_version_data else None,
                    "last_updated": max(
                        [v["created_at"] for v in model_data["versions"].values()],
                        default=model_data["created_at"]
                    )
                }
                models.append(model_summary)
            
            return sorted(models, key=lambda x: x["last_updated"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise
    
    async def promote_model(
        self,
        model_name: str,
        version_id: str,
        stage: str = "production"
    ) -> Dict[str, Any]:
        """Promote a model version to a specific stage.
        
        Args:
            model_name: Name of the model
            version_id: Version ID to promote
            stage: Target stage (e.g., 'staging', 'production')
            
        Returns:
            Promotion information
        """
        try:
            registry_data = self._load_registry()
            
            if model_name not in registry_data["models"]:
                raise ValueError(f"Model {model_name} not found")
            
            if version_id not in registry_data["models"][model_name]["versions"]:
                raise ValueError(f"Version {version_id} not found")
            
            # Update version status
            for vid, version_data in registry_data["models"][model_name]["versions"].items():
                if vid == version_id:
                    version_data["status"] = stage
                elif version_data["status"] == stage:
                    version_data["status"] = "archived"
            
            # Update active version if promoting to production
            if stage == "production":
                registry_data["models"][model_name]["active_version"] = version_id
            
            registry_data["last_updated"] = datetime.now().isoformat()
            self._save_registry(registry_data)
            
            # Promote in MLflow if available
            try:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version_id,
                    stage=stage.title()
                )
            except Exception as e:
                logger.warning(f"Failed to promote in MLflow: {e}")
            
            logger.info(f"Model {model_name} version {version_id} promoted to {stage}")
            
            return {
                "model_name": model_name,
                "version_id": version_id,
                "new_stage": stage,
                "promoted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    async def archive_model_version(self, model_name: str, version_id: str) -> Dict[str, Any]:
        """Archive a model version.
        
        Args:
            model_name: Name of the model
            version_id: Version ID to archive
            
        Returns:
            Archive information
        """
        try:
            registry_data = self._load_registry()
            
            if model_name not in registry_data["models"]:
                raise ValueError(f"Model {model_name} not found")
            
            if version_id not in registry_data["models"][model_name]["versions"]:
                raise ValueError(f"Version {version_id} not found")
            
            # Update status
            registry_data["models"][model_name]["versions"][version_id]["status"] = "archived"
            registry_data["models"][model_name]["versions"][version_id]["archived_at"] = datetime.now().isoformat()
            
            # If this was the active version, find the next best version
            if registry_data["models"][model_name]["active_version"] == version_id:
                new_active = self._find_best_available_version(registry_data["models"][model_name]["versions"])
                registry_data["models"][model_name]["active_version"] = new_active
            
            registry_data["last_updated"] = datetime.now().isoformat()
            self._save_registry(registry_data)
            
            logger.info(f"Model {model_name} version {version_id} archived")
            
            return {
                "model_name": model_name,
                "version_id": version_id,
                "archived_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to archive model version: {e}")
            raise
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry data from file."""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {"models": {}, "deployments": {}}
    
    def _save_registry(self, data: Dict[str, Any]) -> None:
        """Save registry data to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise
    
    def _is_better_model(self, new_version: Dict[str, Any], current_version: Optional[Dict[str, Any]]) -> bool:
        """Determine if a new version is better than the current one."""
        if current_version is None:
            return True
        
        # Compare based on MAE (lower is better)
        new_mae = new_version.get("metrics", {}).get("mae", float('inf'))
        current_mae = current_version.get("metrics", {}).get("mae", float('inf'))
        
        return new_mae < current_mae
    
    def _find_best_available_version(self, versions: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Find the best available (non-archived) version."""
        available_versions = {
            vid: vdata for vid, vdata in versions.items()
            if vdata.get("status") != "archived"
        }
        
        if not available_versions:
            return None
        
        # Return version with lowest MAE
        best_version = min(
            available_versions.keys(),
            key=lambda x: available_versions[x].get("metrics", {}).get("mae", float('inf'))
        )
        
        return best_version
    
    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status.
        
        Returns:
            Deployment status information
        """
        try:
            registry_data = self._load_registry()
            
            active_deployments = {}
            for deployment_id, deployment_data in registry_data["deployments"].items():
                if deployment_data["status"] == "deployed":
                    target = deployment_data["target"]
                    active_deployments[target] = deployment_data
            
            return {
                "active_deployments": active_deployments,
                "total_deployments": len(registry_data["deployments"]),
                "last_updated": registry_data.get("last_updated"),
                "registry_status": "healthy"
            }
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            raise
