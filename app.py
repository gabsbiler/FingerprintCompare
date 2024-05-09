import os
import cv2
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Configuration
UPLOAD_FOLDER = 'fingerprints'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
SECRET_KEY = 'dev'

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = SECRET_KEY

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_fingerprint_match_score():
    try:
        # Load images
        path1 = os.path.join(app.config['UPLOAD_FOLDER'], "fingerprint_1.jpeg")
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], "fingerprint_2.jpeg")
        fingerprint1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        fingerprint2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        keypoints_1, des1 = sift.detectAndCompute(fingerprint1, None)
        keypoints_2, des2 = sift.detectAndCompute(fingerprint2, None)
        
        # Matcher and match points
        matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {})
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        if len(keypoints_1) == 0 or len(keypoints_2) == 0:
            return 0  # Avoid division by zero
        match_score = len(good_matches) / min(len(keypoints_1), len(keypoints_2)) * 100
        return match_score
    except Exception as e:
        print(f"Error processing fingerprint images: {e}")
        return 0

@app.route('/verify/fingerprint', methods=['POST'])
def verify_fingerprint():
    if 'file' not in request.files:
        abort(400, description="No file part")
    files = request.files.getlist('file')
    if len(files) != 2:
        abort(400, description="Exactly two files required")
    
    # Save and secure the files
    for idx, file in enumerate(files):
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], f"fingerprint_{idx+1}.jpeg"))
        else:
            abort(400, description="Unsupported file type or invalid file")
    
    # Calculate match score and clean up
    match_score = get_fingerprint_match_score()
    for idx in range(1, 3):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f"fingerprint_{idx}.jpeg"))
    
    return jsonify({
        "status": "success",
        "message": "Verification completed successfully",
        "match_score": match_score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
