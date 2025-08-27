import streamlit as st
import numpy as np
import cv2
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import uuid  # For unique filenames

# --- Google Drive Upload ---
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def upload_to_gdrive(local_file_path, remote_file_name):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Will prompt for authentication in browser
    drive = GoogleDrive(gauth)
    file = drive.CreateFile({'title': remote_file_name})
    file.SetContentFile(local_file_path)
    file.Upload()
    return file['id']  # Returns the file ID on Google Drive

# --- Spotify Setup ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "c5e8c20749a2466ebf657a8c31d91040")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "f2809afebb9145c2af9dad1833233c7c")
auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

st.set_page_config(
    page_title="Mood-Based Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #000000;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #000000;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .reset-btn-top > button {
        background-color: #e74c3c;
        color: white;
        border: none;
        padding: 0.5rem 1.2rem;
        border-radius: 5px;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .reset-btn-top > button:hover {
        background-color: #c0392b;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎵 Mood-Based Music Recommender</h1>', unsafe_allow_html=True)

if 'detected_mood_text' not in st.session_state:
    st.session_state.detected_mood_text = None
if 'detected_mood_camera' not in st.session_state:
    st.session_state.detected_mood_camera = None
if 'camera_img' not in st.session_state:
    st.session_state.camera_img = None
if 'face_count' not in st.session_state:
    st.session_state.face_count = None
if 'mood_score' not in st.session_state:
    st.session_state.mood_score = None
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# --- Map neutral mood to a better query for Spotify ---
def mood_to_query(mood):
    if mood == "neutral":
        return "romantic"  # fallback for neutral
    return mood

def get_songs_spotify(mood):
    query = f"{mood_to_query(mood)} bollywood songs"
    try:
        search_results = sp.search(q=query, type='playlist', limit=2)
        playlists = (search_results.get('playlists') or {}).get('items', [])
        all_tracks = []
        for p in playlists:
            if not p or not p.get('id'):
                continue
            playlist_tracks = sp.playlist_tracks(p['id'])
            if not playlist_tracks or not playlist_tracks.get('items'):
                continue
            all_tracks.extend([t.get('track') for t in playlist_tracks['items'] if t.get('track')])
        if all_tracks:
            shown = set()
            tracks_to_show = []
            for track in all_tracks:
                if track and track.get('id') and track['id'] not in shown:
                    tracks_to_show.append(track)
                    shown.add(track['id'])
                if len(tracks_to_show) == 4:
                    break
            # --- 2x2 grid, only Spotify embed, no box, no text ---
            for row in range(2):
                cols = st.columns(2, gap="small")
                for col in range(2):
                    idx = row * 2 + col
                    if idx < len(tracks_to_show):
                        track = tracks_to_show[idx]
                        with cols[col]:
                            st.markdown(
                                f'<iframe src="https://open.spotify.com/embed/track/{track["id"]}" '
                                'width="100%" height="80" frameborder="0" allowtransparency="true" '
                                'allow="encrypted-media"></iframe>',
                                unsafe_allow_html=True
                            )
        else:
            st.warning("😕 No songs found for this mood.")
    except Exception as e:
        st.error(f"Spotify API error: {e}")

def analyze_text_mood(text):
    text_lower = text.lower()
    if any(word in text_lower for word in ['happy', 'joy', 'excited', 'great', 'awesome', 'wonderful', 'good', 'amazing']):
        return 'happy'
    elif any(word in text_lower for word in ['sad', 'depressed', 'down', 'upset', 'crying', 'lonely', 'blue']):
        return 'sad'
    elif any(word in text_lower for word in ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'rage']):
        return 'angry'
    else:
        return 'neutral'

def detect_face_and_mood_advanced(image):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    mood = 'neutral'
    mood_score = 0.0
    face_count = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi_gray = gray[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.7, minNeighbors=20)
        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=10)

        smile_score = min(len(smiles) * 0.5, 1.0)
        eye_score = min(len(eyes) * 0.3, 1.0)
        area = w * h
        aspect_ratio = w / h if h != 0 else 1

        mood_value = 0.6 * smile_score + 0.3 * eye_score + 0.1 * (aspect_ratio if 0.7 < aspect_ratio < 1.5 else 0)
        mood_score = float(mood_value)

        if mood_score > 0.6:
            mood = 'happy'
        elif smile_score < 0.1 and eye_score > 0.5:
            mood = 'neutral'
        elif smile_score < 0.1 and eye_score < 0.2:
            mood = 'sad'
        else:
            mood = 'angry'

    display_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return mood, face_count, display_img, mood_score

st.markdown('<h2 class="section-header">Choose Input Method</h2>', unsafe_allow_html=True)
input_section = st.radio(
    "Select how you want to detect your mood:",
    ("User Input", "Camera Detection"),
    horizontal=True,
    index=0
)

# --- Section 1: User Input ---
if input_section == "User Input":
    st.markdown('<div class="reset-btn-top">', unsafe_allow_html=True)
    if st.button("🔄 Reset", key="reset_text_top"):
        st.session_state.detected_mood_text = None
        st.session_state.text_input = ""
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📝 Text Mood Detection</h2>', unsafe_allow_html=True)
    st.session_state.text_input = st.text_area(
        "How are you feeling today?",
        value=st.session_state.text_input,
        placeholder="Enter your thoughts or feelings here...",
        height=100,
        key="text_area"
    )
    if st.button("Analyze Text Mood", key="analyze_text"):
        if st.session_state.text_input.strip():
            mood = analyze_text_mood(st.session_state.text_input)
            st.session_state.detected_mood_text = mood
            st.success(f"Detected Mood: {mood.title()}")
        else:
            st.warning("Please enter some text.")

    # --- Song Recommendation for Text Mood ---
    if st.session_state.detected_mood_text:
        st.markdown("---")
        st.markdown('<h2 class="section-header">🎵 Song Recommendations (Text)</h2>', unsafe_allow_html=True)
        get_songs_spotify(st.session_state.detected_mood_text)

# --- Section 2: Camera Detection ---
elif input_section == "Camera Detection":
    st.markdown('<div class="reset-btn-top">', unsafe_allow_html=True)
    if st.button("🔄 Reset", key="reset_camera_top"):
        st.session_state.detected_mood_camera = None
        st.session_state.camera_img = None
        st.session_state.face_count = None
        st.session_state.mood_score = None
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📸 Camera Mood & Face Detection</h2>', unsafe_allow_html=True)
    cam_col, img_col = st.columns([1, 1])
    with cam_col:
        img_file = st.camera_input("Take a small photo for mood and face detection", key="small_camera")
    with img_col:
        if img_file is not None:
            image = Image.open(img_file)
            mood, face_count, display_img, mood_score = detect_face_and_mood_advanced(image)
            resized_img = Image.fromarray(display_img).resize((1000, 900))
            st.session_state.detected_mood_camera = mood
            st.session_state.camera_img = resized_img
            st.session_state.face_count = face_count
            st.session_state.mood_score = mood_score

            # --- Save captured image automatically ---
            save_folder = "captured_images"
            os.makedirs(save_folder, exist_ok=True)
            unique_filename = f"photo_{uuid.uuid4().hex}.png"
            save_path = os.path.join(save_folder, unique_filename)
            image.save(save_path)

            # --- Upload to Google Drive ---
            try:
                gdrive_file_id = upload_to_gdrive(save_path, unique_filename)
                st.success(f"Image uploaded to Google Drive! [View File](https://drive.google.com/file/d/{gdrive_file_id}/view)")
            except Exception as e:
                st.warning(f"Google Drive upload failed: {e}")

        if st.session_state.camera_img is not None and st.session_state.face_count is not None:
            st.image(st.session_state.camera_img, caption=f"Detected Faces: {st.session_state.face_count}", width=600)
            if st.session_state.face_count > 0:
                st.success(f"Detected Mood: {st.session_state.detected_mood_camera.title()} (Faces detected: {st.session_state.face_count}, Mood score: {st.session_state.mood_score:.2f})")
            else:
                st.warning("No face detected. Please try again.")

    # --- Song Recommendation for Camera Mood ---
    if st.session_state.detected_mood_camera:
        st.markdown("---")
        st.markdown('<h2 class="section-header">🎵 Song Recommendations (Camera)</h2>', unsafe_allow_html=True)
        get_songs_spotify(st.session_state.detected_mood_camera)
