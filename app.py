import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from tensorflow.keras.preprocessing import image
from googleapiclient.discovery import build

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests (necessary for frontend-backend communication)

# Load your trained model
model = tf.keras.models.load_model('emotiondetector.h5')


# Set up YouTube API
YOUTUBE_API_KEY = 'AIzaSyBLeiHS_n7iKhDLchAwCsSYYaogzSpuHaA'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Emotion mapping
emotion_labels = ["happy", "sad", "angry", "surprise", "neutral"]

# Function to handle emotion prediction
def predict_emotion(img):
    img = image.load_img(img, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions[0])
    return emotion_labels[emotion_index]

# Function to fetch music suggestions based on emotion
def get_youtube_playlist(emotion):
    playlists = {
        'happy': 'happy music playlist',
        'sad': 'sad music playlist',
        'angry': 'angry music playlist',
        'surprise': 'surprise music playlist',
        'neutral': 'calm music playlist'
    }

    query = playlists.get(emotion, 'happy music playlist')  # Default to happy music
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="playlist",
        maxResults=5
    )
    response = request.execute()

    playlist_links = []
    for item in response['items']:
        playlist_links.append(f"https://www.youtube.com/playlist?list={item['id']['playlistId']}")

    return playlist_links

# Endpoint for detecting emotion and fetching music suggestion
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    # Read image from request
    image_data = request.files['image']
    
    # Save image temporarily for processing
    img = image_data.read()
    img = BytesIO(img)
    
    # Predict emotion
    emotion = predict_emotion(img)
    
    # Fetch playlist based on emotion
    playlist = get_youtube_playlist(emotion)

    # Return emotion and playlist as response
    return jsonify({
        'emotion': emotion,
        'playlist': playlist
    })

if __name__ == '__main__':
    app.run(debug=True)
