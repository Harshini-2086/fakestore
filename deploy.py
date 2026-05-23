import subprocess
import sys

def run_command(command, description):
    print(f"\n🚀 Running: {description}...")
    try:
        subprocess.run(command, shell=True, check=True, text=True)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during: {description}")
        print(f"Details: {e}")
        sys.exit(1)

def main():
    print("=== STARTING GOOGLE CLOUD AUTOMATED DEPLOYMENT ===")
    
    # 1. Point to your unique Google Cloud Project ID
    run_command("gcloud config set project fastapi-fresh-deploy-2026-3", "Setting active project ID")
    
    # 2. Tie the project to your billing configurations 
    # (Note: If this fails due to IAM permissions, link billing manually in GCP Console and comment this out)
    run_command(
        "gcloud billing projects link fastapi-fresh-deploy-2026-3 --billing-account=0197EE-C0ABE4-83D756", 
        "Linking billing profile"
    )
    
    # 3. Turn on the necessary Google Cloud machinery APIs
    run_command(
        "gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com", 
        "Enabling Google Cloud APIs"
    )

    # 4. Deploy code (Cloud Run reads the Dockerfile automatically; no messy --command string needed)
    deploy_cmd = "gcloud run deploy fake-store-api --source . --allow-unauthenticated --region us-central1 --quiet"
    run_command(deploy_cmd, "Deploying application to Cloud Run")

    print("\n🎉 ALL DONE! Your FastAPI server deployment script finished perfectly!")

if __name__ == "__main__":
    main()