Project Overview: Music Genre Classification

This project is a Music Genre Classification Web Application built using React (Frontend) and Flask (Backend). The application allows users to upload an audio file, processes it using a trained deep learning model, and predicts the genre of the music.

1. Project Structure
   
   ReactProjects/
│   ├── src/
│   │   ├── App.js         # Main React Component
│   │   ├── App.css        # Styles
│   │   ├── index.js       # Entry Point
│   ├── public/
│   ├── package.json       # React dependencies
│   ├── .gitignore
│── backend/               # Flask Backend
│   ├── app.py             # Flask API for Genre Prediction
│   ├── requirements.txt   # Python Dependencies
│   ├── model/             # Pretrained Model & Assets
│   │   ├── scaler.pkl     
│   │   ├── pca.pkl
│   │   ├── music_classifier.pkl
│   │   ├── label_encoder.pkl
│── Training Scripts/               # Jupyter notebook file with training model scripts
│── README.md              # Project Documentation

3. Features

	•	 Upload an audio file from the frontend.
	•	🔥 Backend processes the file by extracting features.
	•	🎯 Pre-trained deep learning model predicts the music genre.
	•	📊 Displays the predicted genre on the UI.

3. Technologies Used

Frontend (React)

	•	React.js (Functional Components, Hooks)
	•	Fetch API (for making HTTP requests)
	•	CSS (for styling)

Backend (Flask)

	•	Flask (for API)
	•	librosa (for audio feature extraction)
	•	torch (PyTorch for deep learning model)
	•	numpy, pandas (data processing)
	•	pickle (for loading pre-trained model)

 4. How It Works

Step 1: User Uploads an Audio File

	•	The React frontend provides a file input to upload an audio file.
	•	The file is sent to the Flask backend using a POST request.

Step 2: Backend Processes the Audio File

	•	Extracts features from the audio using Librosa.
	•	Scales and applies Principal Component Analysis (PCA).
	•	Runs the deep learning model (pre-trained in PyTorch).
	•	Predicts the music genre.

Step 3: Returns the Prediction to the Frontend

	•	The predicted genre is sent as a JSON response.
	•	The frontend updates the UI to display the genre.
