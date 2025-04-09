import yaml
import mlflow
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")

with open("./constants.yaml", "r") as f:
            const_data = yaml.safe_load(f)

model_name= const_data["model_name"]


def promote_staging_to_production():
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        # Get current staging version
        staging_versions = [v for v in versions if v.tags.get("version_status").lower() == "staging"]
        if not staging_versions:
            raise ValueError("No model version tagged as 'staging'.")
        
        staging_model = staging_versions[0]
        staging_version = staging_model.version

        # Promote staging to production
        client.set_model_version_tag(
            name=model_name,
            version=staging_version,
            key="version_status",
            value="production"
        )
        print(f"Promoted model version {staging_version} to 'production'.")

        # Archive previous production versions
        production_versions = [
            v for v in versions
            if v.tags.get("version_status") == "production"
            and v.version != staging_version
        ]

        for v in production_versions:
            client.delete_model_version_tag(
                name=model_name,
                version=v.version,
                key="version_status"
            )
            print(f"Archived model version {v.version} (previous production)")

        return staging_version

    except Exception as e :
         print(f"Error occurred from staging to production : {e}")

if __name__ == "__main__":
    promote_staging_to_production()