# MLOps Project - House Price Prediction (Student ID: 23L2504)

A simple, reproducible MLOps pipeline that separates source code, data, and model
artifacts, using Git and GitHub for version control.

## Project Structure

```
├── data/                          # Raw dataset (ignored by git)
├── src/
│   └── train_23L2504.py      # Training script
├── model/                         # Trained model output (ignored by git)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/<your-username>/mlops-project-23L2504.git
   cd mlops-project-23L2504
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Place your dataset as `data/dataset.csv` (not tracked by git).

## Running the Training Script

From the project root directory:

```
python src/train_23L2504.py
```

This will:
- Load `data/dataset.csv`
- Train a RandomForest regression model
- Save the trained model to `model/model_23L2504.pkl`

## Notes

- Raw data and trained model files are intentionally excluded from version control (see `.gitignore`).
- Only source code (`.py`) and configuration files (`.txt`, `.md`, `.gitignore`) are committed.
