import subprocess
import os
import sys

def run_command(command):
    """Runs a command and handles errors."""
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(command)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    print(result.stdout)
    return result

def train_model():
    """Trains a new model and returns the path to the saved regressor."""
    print("\\n" + "="*20)
    print("Step 1: Training the model")
    print("="*20)
    outdir = "runs/pipeline_experiment"
    model_name = "pipeline_model_v1"
    regressor_path = os.path.join(outdir, f"ridge_regressor_{model_name}.pkl")

    command = [
        "python3",
        "shorts_model/modeling/train.py",
        "--csv", "data/processed/training-data_v4.3_with-pseudo.csv",
        "--outdir", outdir,
        "--name", model_name,
    ]
    run_command(command)
    print(f"Model trained and saved to {regressor_path}")
    return regressor_path

def run_inference(regressor_path):
    """Runs inference using the trained model."""
    print("\\n" + "="*20)
    print("Step 2: Running inference")
    print("="*20)
    outdir = "inference/pipeline_inference"

    command = [
        "python3",
        "shorts_model/inference/infer_minilm_v1.py",
        "--transcript", "data/raw/transcript_anne-applebaum.txt",
        "--regressor_path", regressor_path,
        "--guest", "Anne Applebaum",
        "--top_k", "5",
        "--outdir", outdir,
    ]
    run_command(command)
    print(f"Inference report saved in {outdir}")

def generate_dependency_graph():
    """Generates a dependency graph for the shorts_model."""
    print("\\n" + "="*20)
    print("Step 3: Generating dependency graph")
    print("="*20)
    output_file = "reports/dependency_graph.svg"

    # Ensure pydeps is installed
    try:
        import pydeps
    except ImportError:
        print("pydeps not found. Installing...")
        run_command([sys.executable, "-m", "pip", "install", "pydeps"])

    command = ["pydeps", "-o", output_file, "shorts_model"]
    run_command(command)
    print(f"Dependency graph saved to {output_file}")


if __name__ == "__main__":
    print("Starting the reproducible pipeline...")

    # Step 1: Train the model
    trained_regressor_path = train_model()

    # Step 2: Run inference
    run_inference(trained_regressor_path)

    # Step 3: Generate dependency graph
    generate_dependency_graph()

    print("\\nPipeline completed successfully!")
