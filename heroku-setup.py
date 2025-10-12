#!/usr/bin/env python3
"""
Heroku deployment setup script for NaijaEstateAI.

This script helps ensure all necessary files are created and
provides deployment instructions.
"""
import os
import sys
from pathlib import Path


def check_heroku_files():
    """Check if all required Heroku deployment files exist."""
    required_files = {
        'requirements.txt': 'Python dependencies',
        'Procfile': 'Heroku process definition',
        '.python-version': 'Python version specification',
    }

    missing_files = []

    for file_name, description in required_files.items():
        if not Path(file_name).exists():
            missing_files.append(f"{file_name} ({description})")

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ All required Heroku deployment files present!")
        return True


def print_deployment_instructions():
    """Print step-by-step deployment instructions."""
    print("\n🚀 Heroku Deployment Instructions:")
    print("\n1. Install Heroku CLI:")
    print("   Download from: https://devcenter.heroku.com/articles/heroku-cli")

    print("\n2. Login to Heroku:")
    print("   heroku login")

    print("\n3. Create a new Heroku app:")
    print("   heroku create your-naijaestateai-backend")

    print("\n4. Set environment variables (optional):")
    print("   heroku config:set MODEL_VERSION=1.0.0")
    print("   heroku config:set ENABLE_METRICS=true")

    print("\n5. Deploy to Heroku:")
    print("   git add .")
    print("   git commit -m 'Deploy to Heroku'")
    print("   git push heroku main")

    print("\n6. Open your deployed app:")
    print("   heroku open")

    print("\n7. View logs (if needed):")
    print("   heroku logs --tail")

    print("\n8. Your API will be available at:")
    print("   https://your-naijaestateai-backend.herokuapp.com")
    print("   API docs: https://your-naijaestateai-backend.herokuapp.com/docs")


def main():
    print("NaijaEstateAI Heroku Deployment Setup")
    print("=" * 40)

    if check_heroku_files():
        print_deployment_instructions()

        print("\n📝 Next Steps:")
        print("1. Make sure you have trained your model (run: python train.py)")
        print("2. Test locally first (run: uvicorn api_app:app --reload)")
        print("3. Follow the deployment instructions above")
        print("4. Update your frontend to use the new Heroku URL")

    else:
        print("\n❌ Please create the missing files before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
