import cv2
import os
from tqdm import tqdm

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def extract_faces(video_path, output_dir, max_frames=10):
    """Extract faces from video"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    faces_saved = []
    frame_count = 0
    faces_extracted = 0
    
    video_name = os.path.basename(video_path).split('.')[0]
    
    while cap.isOpened() and faces_extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 10 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                face_img = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, (224, 224))
                
                face_filename = f"{video_name}_face_{faces_extracted}.jpg"
                face_path = os.path.join(output_dir, face_filename)
                cv2.imwrite(face_path, face_resized)
                
                faces_saved.append(face_path)
                faces_extracted += 1
        
        frame_count += 1
    
    cap.release()
    return faces_saved

def process_video_dataset(video_dir, output_dir, label, max_videos=200):
    """Process all videos"""
    label_output_dir = os.path.join(output_dir, label)
    os.makedirs(label_output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files = video_files[:max_videos] if max_videos else video_files
    
    print(f"\nProcessing {len(video_files)} {label} videos...")
    
    all_faces = []
    for video_file in tqdm(video_files):
        video_path = os.path.join(video_dir, video_file)
        try:
            faces = extract_faces(video_path, label_output_dir, max_frames=10)
            all_faces.extend(faces)
        except Exception as e:
            print(f"Error with {video_file}: {e}")
    
    print(f"Total {label} faces: {len(all_faces)}")
    return all_faces

if __name__ == "__main__":
    BASE_DIR = "./data"
    OUTPUT_DIR = "./processed_faces"
    
    print("=" * 50)
    print("EXTRACTING REAL FACES")
    print("=" * 50)
    real_video_dir = os.path.join(BASE_DIR, "original_sequences/youtube/c23/videos")
    real_faces = process_video_dataset(real_video_dir, OUTPUT_DIR, 'real', max_videos=200)
    
    print("\n" + "=" * 50)
    print("EXTRACTING FAKE FACES")
    print("=" * 50)
    fake_video_dir = os.path.join(BASE_DIR, "manipulated_sequences/Deepfakes/c23/videos")
    fake_faces = process_video_dataset(fake_video_dir, OUTPUT_DIR, 'fake', max_videos=200)
    
    print("\n" + "=" * 50)
    print("✅ DONE!")
    print("=" * 50)
    print(f"Real faces: {len(real_faces)}")
    print(f"Fake faces: {len(fake_faces)}")