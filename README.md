# 🪁 PneumoScan AI

An advanced deep learning-powered web application for pneumonia detection from chest X-rays using Streamlit and PyTorch.

## 🌟 Features

- Real-time pneumonia detection from chest X-rays
- Interactive visualization with feature heatmaps
- Confidence meter for prediction reliability
- Detailed analysis reports
- Historical tracking of detections
- Responsive design with modern UI
- Comprehensive statistics dashboard

## 🚀 Live Demo
[pneumoscan-ai-fyycvjlm6hh7xqhzasepey.streamlit.app/](https://pneumoscan-ai-fyycvjlm6hh7xqhzasepey.streamlit.app/)

## 🛠️ Tech Stack

- Python 3.8+
- Streamlit
- PyTorch & TorchVision
- OpenCV
- Plotly
- Pandas & NumPy
- PIL (Python Imaging Library)

## 👌 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/arpanksasmal/PneumoScan-AI.git
cd PneumoScan-AI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 💽 Project Structure

```
pneumoscan-ai/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── style.css              # Custom CSS styles
├── model/
│   └── best_pneumonia_model_densenet121.pt  # Trained model weights
├── detection_history/     # Folder for storing detection logs
├── test_data/             # Sample chest X-ray images for testing
│   ├── sample_1.jpg
│   ├── sample_2.jpg
│   ├── sample_3.jpg
|   ├── sample_4.jpg
└── README.md             # Project documentation
```

## 🚀 Running the Application

1. Ensure you're in the project directory and virtual environment is activated
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. Open your browser and navigate to `http://localhost:8501`

## 📊 Model Architecture

The application uses a modified DenseNet121 architecture with:
- Custom classifier layers for binary classification
- Batch normalization and dropout for regularization
- Pre-trained weights for feature extraction

## 📊 Performance Metrics

Our model achieves the following performance on the test set:

- Accuracy: 87.82%
- Precision: 90.00%
- Recall: 88.00% 
- F1 Score: 87.00% 
- AUC-ROC: 96.87%

Cross-validation results show consistent performance across different data splits, with a standard deviation of less than 1.5% for all metrics.

## 🔐 Security Considerations

- The application is designed for testing and educational purposes
- Not HIPAA compliant by default
- Medical data should be handled according to relevant privacy regulations
- Use appropriate security measures when deploying in production

## 🌐 Deployment

### Local Deployment
Follow the installation instructions above.

### Cloud Deployment (Recommended Platforms)
- **Streamlit Cloud** (Easiest)
  1. Push code to GitHub
  2. Connect repository to Streamlit Cloud
  3. Configure requirements.txt
  4. Deploy

- **Heroku**
  1. Create Procfile:
     ```
     web: streamlit run app.py
     ```
  2. Configure runtime.txt with Python version
  3. Deploy using Heroku CLI or GitHub integration

- **AWS/GCP/Azure**
  - Use container services (ECS, GKE, AKS)
  - Configure environment variables
  - Set up load balancers if needed

## 📕 Test Data Instructions

To help users verify model predictions, we provide sample chest X-ray images in the `test_data/` folder.

1. Navigate to the `test_data/` folder in your project directory.
2. Choose an image (e.g., `sample_1.jpg`).
3. Upload the image using the Streamlit web app.
4. Click "Predict Pneumonia" to see the results.

This ensures a **consistent** and **easy testing experience** for users.

## ⚠️ Important Notes

- This tool is for screening purposes only
- Not a replacement for professional medical diagnosis
- Consult healthcare providers for medical decisions
- Regular model updates recommended for optimal performance

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👍 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- Special thanks to **OpenAI** and **Deep Learning Research Community**
- Dataset sourced from [Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## 📄 Conclusion

This project serves as an advanced AI-driven tool for pneumonia detection. While it provides high accuracy and real-time predictions, it is intended for **educational and research purposes** only. Always consult medical professionals for official diagnoses.

## 📧 Contact

- **Arpan Kumar Sasmal**
- LinkedIn: [www.linkedin.com/in/arpan-kumar-sasmal-2b4421240](https://www.linkedin.com/in/arpan-kumar-sasmal-2b4421240)
- Email: [arpankumarsasmal@gmail.com](mailto:arpankumarsasmal@gmail.com)

---
Created with ❤️ by **Arpan Kumar Sasmal** 😎
