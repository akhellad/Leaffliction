from invoke import task

PYTHON = "python3"
PLANTS = {
    "Apple": ["Apple_Black_rot", "Apple_healthy", "Apple_rust", "Apple_scab"],
    "Grape": ["Grape_Black_rot", "Grape_Esca", "Grape_healthy", "Grape_spot"],
}


@task
def clean(c):
    """Remove Python cache files."""
    c.run('find . -type d -name "__pycache__" -exec rm -rf {} + '
          '2>/dev/null || true')
    c.run('find . -type f -name "*.pyc" -delete 2>/dev/null || true')


@task
def distribution(c, plant="Apple"):
    """Run distribution analysis."""
    c.run(f"{PYTHON} scripts/Distribution.py {plant}/")


@task
def augmentation(c, plant="Apple"):
    """Balance/augment dataset."""
    c.run(f"{PYTHON} scripts/Augmentation.py {plant}/")


@task
def transformation(c, plant="Apple"):
    """Apply transformations."""
    for subdir in PLANTS.get(plant, []):
        c.run(f"mkdir -p transformed/{subdir}")
        c.run(f"{PYTHON} scripts/Transformation.py -src {plant}/{subdir} "
              f"-dst transformed/{subdir}")


@task
def train(c, plant="Apple", epochs=50, batch_size=32):
    """Train CNN model."""
    c.run(f"{PYTHON} scripts/train.py {plant}/ {epochs} "
          f"--batch-size {batch_size}")


@task
def predict(c, image, model="output_model.zip", data_dir="Apple/"):
    """Run inference."""
    c.run(f'{PYTHON} scripts/predict.py -m {model} -d {data_dir} "{image}"')


@task
def lint(c):
    """Run flake8 linter."""
    c.run(f"{PYTHON} -m flake8 *.py scripts/ "
          f"--show-source --statistics")


@task
def pipeline(c, plant="Apple"):
    """Run full pipeline: distribution, augmentation, transformation, train."""
    distribution(c, plant)
    augmentation(c, plant)
    transformation(c, plant)
    train(c, plant)
