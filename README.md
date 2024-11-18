# ML Model CI/CD Pipeline

This project demonstrates a CI/CD pipeline for a simple CNN model trained on MNIST dataset.

## Project Structure 
.
├── model/
│ └── train.py

├── tests/

│ └── test_model.py

├── .github/

│ └── workflows/

│ └── ml-pipeline.yml

├── requirements.txt

└── README.md

## Local Setup

1. Create a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

2. Install dependencies:
pip install -r requirements.txt

3. Train the model:
python model/train.py

4. Run tests:
python tests/test_model.py

## CI/CD Pipeline

The GitHub Actions pipeline will automatically:
1. Set up Python environment
2. Install dependencies
3. Train the model
4. Run tests to verify:
   - Model architecture (input/output dimensions)
   - Parameter count (< 25000)
   - Model accuracy (> 95%)
5. Archive the trained model as an artifact

## Model Details

- Architecture: Simple CNN with 2 convolutional layers and 2 fully connected layers
- Dataset: MNIST
- Training: 1 epoch
- Input shape: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## GitHub Setup

1. Create a new repository on GitHub
2. Initialize git in your local project:
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main


The GitHub Actions pipeline will automatically run on every push to the repository.

To use this project:

Create all the files in your local directory with the exact structure shown in the README.md
Create a virtual environment and install dependencies
Run the training script locally to ensure it works
Run the tests locally to verify everything passes
Create a GitHub repository and push the code
Check the Actions tab on GitHub to see the pipeline running

The pipeline will:
1. Train the model
2. Verify the model architecture
3. Check parameter count
4. Validate model accuracy
5. Save the model with a timestamp
6. Archive the model as an artifact

The tests specifically check:
1. Model has less than 25000 parameters
2. Model accepts 28x28 input
3. Model outputs 10 classes
4. Model achieves >95% accuracy

Each time you train the model, it will be saved with a timestamp suffix for tracking purposes.

