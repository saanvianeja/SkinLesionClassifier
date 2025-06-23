# Skin Lesion Classifier (Flask)

A comprehensive medical analysis tool for skin lesion assessment using advanced computer vision (CNN) and medical criteria (ABCDE, risk factors).

## Features

- **Image Upload & Analysis**: Upload skin lesion images for detailed analysis
- **ABCDE Criteria**: Evaluates asymmetry, border irregularity, color variation, diameter, and evolution
- **Skin Type Integration**: Fitzpatrick skin type classification for improved accuracy
- **Risk Assessment**: Combines CNN model, ABCDE, and metadata (age, UV, family history, body part, evolution)
- **Confidence Scores**: Shows percent confidence in malignancy (0% = benign, 100% = malignant)
- **Medical Guidelines**: Follows established dermatological assessment protocols
- **Explanations**: UI displays explanations for each analysis factor

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Image Processing**: OpenCV, PIL
- **Machine Learning**: PyTorch, scikit-learn
- **Medical Analysis**: Custom ABCDE algorithm implementation

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saanvianeja/Flask-V.2.git
   cd Flask-V.2
   ```

2. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt  # or install from pyproject.toml if provided
   ```

4. **Model Weights**:
   - The CNN model weights (`best_isic_model.pth`) are NOT included in the repository due to size limits.
   - To use the CNN, place your trained `best_isic_model.pth` in the project root.
   - If missing, the app will fall back to metadata-only analysis.

5. **Run the application**:
   ```bash
   python main.py
   ```

6. **Access the application**:
   Open your browser and go to `http://127.0.0.1:5003` (or the port shown in the terminal)

## Usage

1. **Upload Image**: Select a clear image of the skin lesion
2. **Enter Metadata**: Fill in age, UV exposure, family history, body part, and evolution
3. **Get Analysis**: Review the detailed ABCDE, CNN, and risk factor analysis and recommendations

## ISIC Dataset
- This project can use real ISIC images for model training (not included in the repo).
- To train your own model, download ISIC data and use the provided training scripts.
- See `train_real_isic.py` and `improved_cnn_trainer.py` for details.

## Troubleshooting
- **Port already in use**: If you see `Address already in use`, stop other Flask servers or change the port in `main.py`.
- **Missing model file**: If `best_isic_model.pth` is missing, CNN analysis will be disabled.
- **IndentationError**: Ensure all Python files use consistent indentation (spaces only).
- **Other errors**: Check the terminal output for details and ensure all dependencies are installed.

## Medical Disclaimer

This tool is designed for educational and screening purposes only. It should not replace professional medical evaluation, diagnosis, or treatment. Always consult with a qualified dermatologist for definitive medical advice.

## Contributing

Contributions are welcome! Please ensure all medical-related changes follow established clinical guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed by Saanvi Aneja. Maintained and extended for advanced skin lesion analysis. 
