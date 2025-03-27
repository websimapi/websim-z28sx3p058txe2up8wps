import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as tf from '@tensorflow/tfjs';
import * as THREE from 'three';
import { config } from './config.js';

const video = document.getElementById('webcam');
const overlayCanvas = document.getElementById('overlay');
const overlayCtx = overlayCanvas.getContext('2d');
const statusElement = document.getElementById('status');
const debugElement = document.getElementById('debug');
const faceDetectedElement = document.getElementById('face-detected');
const eyesOpenElement = document.getElementById('eyes-open');
const headRotationYElement = document.getElementById('head-rotation-y');
const headRotationXElement = document.getElementById('head-rotation-x');


let model, detector;
let audioContext;
let isGameRunning = false;
let lastDirection = null; // To prevent continuous sound for the same direction

// Debounce function
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Function to play a specific pitch
function playPitch(frequency) {
    if (!config.audioEnabled || !audioContext) return;

    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.type = 'sine'; // sine, square, sawtooth, triangle
    oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
    gainNode.gain.setValueAtTime(config.audioVolume, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, audioContext.currentTime + config.audioDuration);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + config.audioDuration);
}

// Debounced audio functions for each direction
const playLeftTurnSound = debounce(() => playPitch(config.leftTurnPitch), 100);
const playRightTurnSound = debounce(() => playPitch(config.rightTurnPitch), 100);
const playLookUpSound = debounce(() => playPitch(config.lookUpPitch), 100);
const playLookDownSound = debounce(() => playPitch(config.lookDownPitch), 100);


// Function to calculate Eye Aspect Ratio (EAR)
function calculateEAR(eyeLandmarks) {
    // Vertical distances
    const v1 = Math.hypot(eyeLandmarks[1].x - eyeLandmarks[5].x, eyeLandmarks[1].y - eyeLandmarks[5].y);
    const v2 = Math.hypot(eyeLandmarks[2].x - eyeLandmarks[4].x, eyeLandmarks[2].y - eyeLandmarks[4].y);
    // Horizontal distance
    const h = Math.hypot(eyeLandmarks[0].x - eyeLandmarks[3].x, eyeLandmarks[0].y - eyeLandmarks[3].y);

    if (h === 0) return 1.0; // Avoid division by zero, assume open if horizontal distance is zero

    const ear = (v1 + v2) / (2.0 * h);
    return ear;
}

// Function to estimate head rotation using solvePnP
function estimateHeadRotation(landmarks, imageSize) {
    // Simplified 3D model points (approximations) - these should ideally be more accurate
     const modelPoints = [
        { x: 0.0, y: 0.0, z: 0.0 },        // Nose tip (Index 1)
        { x: 0.0, y: -330.0, z: -65.0 },   // Chin (Index 152) - approximate
        { x: -225.0, y: 170.0, z: -135.0 },// Left eye corner (Index 33)
        { x: 225.0, y: 170.0, z: -135.0 }, // Right eye corner (Index 263)
        { x: -150.0, y: -150.0, z: -125.0},// Left mouth corner (Index 61)
        { x: 150.0, y: -150.0, z: -125.0 } // Right mouth corner (Index 291)
    ];

    // Corresponding 2D image points from landmarks
    const imagePoints = [
        landmarks[1],    // Nose tip
        landmarks[152],  // Chin
        landmarks[33],   // Left eye corner
        landmarks[263],  // Right eye corner
        landmarks[61],   // Left mouth corner
        landmarks[291]   // Right mouth corner
    ].map(p => ({ x: p.x * imageSize.width, y: p.y * imageSize.height })); // Scale back to pixel coords

    const focalLength = imageSize.width; // Approximation
    const cameraCenter = { x: imageSize.width / 2, y: imageSize.height / 2 };

    // Camera intrinsic parameters (approximate)
    const cameraMatrix = new THREE.Matrix4();
    cameraMatrix.set(
        focalLength, 0, cameraCenter.x, 0,
        0, focalLength, cameraCenter.y, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    );

    // Convert points to OpenCV format (matrices) - requires a solvePnP implementation or approximation
    // This part is complex without an external library for solvePnP in JS.
    // We'll use a simpler approximation based on eye/nose positions for now.

    // --- Simplified Rotation Estimation ---
    // Use relative positions of key landmarks (e.g., nose, eyes, chin)
    const nose = landmarks[1];
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const chin = landmarks[152];

    // Yaw (Left/Right Turn): Compare horizontal distance from nose to eyes
    const eyeCenterX = (leftEye.x + rightEye.x) / 2;
    const yaw = (nose.x - eyeCenterX); // Positive if turned right, negative if turned left

    // Pitch (Up/Down Look): Compare vertical distance from nose to chin vs nose to eyes
    const eyeCenterY = (leftEye.y + rightEye.y) / 2;
    const pitch = (nose.y - eyeCenterY); // Can refine this logic - using difference for simplicity

    // Roll (Tilt): Compare y-coordinates of left and right eye
    const roll = (rightEye.y - leftEye.y);

    // Convert these simplified metrics to approximate angles (needs calibration)
    // These are not true Euler angles but indicators of movement.
    const rotation = {
        yaw: yaw * Math.PI, // Scale factor - needs adjustment
        pitch: pitch * Math.PI, // Scale factor - needs adjustment
        roll: roll * Math.PI * 2 // Scale factor - needs adjustment
    };

    // Update debug info
    if(config.showDebugInfo) {
        headRotationYElement.textContent = rotation.yaw.toFixed(2);
        headRotationXElement.textContent = rotation.pitch.toFixed(2);
    }

    return rotation; // Return simplified rotation indicators
}


async function detectFace() {
    if (!detector || !video || video.readyState < 3) { // Check if video is ready
        requestAnimationFrame(detectFace);
        return;
    }

    const faces = await detector.estimateFaces(video, {
        flipHorizontal: false, // Flip for webcam view if needed
    });

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height); // Clear previous drawings

    if (faces.length > 0) {
        const face = faces[0];
        const landmarks = face.keypoints; // Use keypoints directly

        if(config.showDebugInfo) faceDetectedElement.textContent = 'true';

        // --- Eye Tracking ---
        const leftEyeLandmarks = [33, 160, 158, 133, 153, 144].map(i => landmarks[i]); // Indices for left eye outline
        const rightEyeLandmarks = [362, 385, 387, 263, 373, 380].map(i => landmarks[i]); // Indices for right eye outline

        const leftEAR = calculateEAR(leftEyeLandmarks);
        const rightEAR = calculateEAR(rightEyeLandmarks);
        const avgEAR = (leftEAR + rightEAR) / 2.0;

        const eyesAreOpen = avgEAR > config.eyeClosedThreshold;

        if(config.showDebugInfo) eyesOpenElement.textContent = eyesAreOpen ? 'true' : 'false';

         // --- Head Pose Estimation ---
        const imageSize = { width: video.videoWidth, height: video.videoHeight };
        const rotation = estimateHeadRotation(landmarks, imageSize); // Get simplified rotation


        // --- Game Logic ---
        if (!eyesAreOpen) {
            if (!isGameRunning) {
                startGame();
            }
             // Handle head rotation only when game is running (eyes closed)
             let currentDirection = null;
             if (rotation.yaw > config.headRotationThresholdY) {
                 currentDirection = 'right';
                 if (lastDirection !== currentDirection) playRightTurnSound();
             } else if (rotation.yaw < -config.headRotationThresholdY) {
                 currentDirection = 'left';
                  if (lastDirection !== currentDirection) playLeftTurnSound();
             } else if (rotation.pitch > config.headRotationThresholdX) { // Adjust pitch detection logic if needed
                 currentDirection = 'down'; // Higher y value usually means looking down in image coords
                  if (lastDirection !== currentDirection) playLookDownSound();
             } else if (rotation.pitch < -config.headRotationThresholdX) {
                  currentDirection = 'up'; // Lower y value usually means looking up
                  if (lastDirection !== currentDirection) playLookUpSound();
             } else {
                 currentDirection = null; // No significant rotation
             }
             lastDirection = currentDirection; // Update last direction

        } else { // Eyes are open
            if (isGameRunning) {
                pauseGame("Eyes opened");
            }
            statusElement.textContent = "Status: Open your eyes or move to pause. Close eyes to play.";
            lastDirection = null; // Reset direction when paused
        }

        // --- Drawing (Optional) ---
        if (config.drawLandmarks) {
            overlayCtx.fillStyle = 'lime';
            landmarks.forEach(point => {
                overlayCtx.beginPath();
                overlayCtx.arc(point.x * overlayCanvas.width, point.y * overlayCanvas.height, 2, 0, 2 * Math.PI);
                overlayCtx.fill();
            });
             // Draw eye outlines
             overlayCtx.strokeStyle = 'red';
             overlayCtx.lineWidth = 1;
             [leftEyeLandmarks, rightEyeLandmarks].forEach(eye => {
                 overlayCtx.beginPath();
                 eye.forEach((p, i) => {
                     const x = p.x * overlayCanvas.width;
                     const y = p.y * overlayCanvas.height;
                     if (i === 0) overlayCtx.moveTo(x, y);
                     else overlayCtx.lineTo(x, y);
                 });
                 overlayCtx.closePath();
                 overlayCtx.stroke();
             });
        }

    } else {
        // No face detected
        if(config.showDebugInfo) faceDetectedElement.textContent = 'false';
        if(config.showDebugInfo) eyesOpenElement.textContent = 'unknown';
        if (isGameRunning) {
            pauseGame("Face lost");
        }
        statusElement.textContent = "Status: Cannot detect face. Please center yourself.";
        lastDirection = null; // Reset direction
    }

    requestAnimationFrame(detectFace);
}

function startGame() {
    console.log("Game Started!");
    isGameRunning = true;
    statusElement.textContent = "Status: Game Running! Move your head.";
    // Initialize or resume game state here
}

function pauseGame(reason = "Paused") {
    console.log(`Game Paused: ${reason}`);
    isGameRunning = false;
    statusElement.textContent = `Status: Paused (${reason}). Close eyes to resume.`;
    lastDirection = null; // Reset direction
    // Pause game logic here
}

async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user' }, // Prefer front camera
            audio: false
        });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                // Set canvas dimensions based on video aspect ratio
                const aspectRatio = video.videoWidth / video.videoHeight;
                const displayWidth = window.innerWidth;
                const displayHeight = window.innerHeight;

                if (displayWidth / displayHeight > aspectRatio) {
                     overlayCanvas.height = displayHeight;
                     overlayCanvas.width = displayHeight * aspectRatio;
                 } else {
                     overlayCanvas.width = displayWidth;
                     overlayCanvas.height = displayWidth / aspectRatio;
                 }
                // Center the canvas
                overlayCanvas.style.left = `${(window.innerWidth - overlayCanvas.width) / 2}px`;
                overlayCanvas.style.top = `${(window.innerHeight - overlayCanvas.height) / 2}px`;

                console.log("Webcam setup complete.");
                resolve();
            };
        });
    } catch (err) {
        console.error("Error accessing webcam:", err);
        statusElement.textContent = "Status: Webcam access denied or not available.";
        alert("Could not access webcam. Please ensure permission is granted and the camera is working.");
    }
}

async function main() {
    statusElement.textContent = "Status: Loading model...";
    await tf.setBackend('webgl'); // Use WebGL backend for performance
    console.log(`Using TensorFlow backend: ${tf.getBackend()}`);

    try {
        model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        detector = await faceLandmarksDetection.createDetector(model, config.modelConfig);
        console.log("Face detection model loaded.");
        statusElement.textContent = "Status: Setting up webcam...";
    } catch (error) {
        console.error("Error loading the model:", error);
        statusElement.textContent = "Status: Error loading face detection model.";
        alert("Failed to load the face detection model. Please check the console for details.");
        return;
    }

    await setupWebcam();

    if (!video.srcObject) return; // Stop if webcam setup failed

    // Setup Audio Context (must be initiated by user action, but loading is ok)
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        // Resume context if needed on first user interaction (implicitly happens when closing eyes)
        if (audioContext.state === 'suspended') {
            const resumeAudio = async () => {
                await audioContext.resume();
                document.body.removeEventListener('click', resumeAudio); // Remove listener after resume
                document.body.removeEventListener('keydown', resumeAudio);
            };
            document.body.addEventListener('click', resumeAudio);
            document.body.addEventListener('keydown', resumeAudio);
        }
    } catch (e) {
        console.error("Web Audio API is not supported in this browser.", e);
        config.audioEnabled = false; // Disable audio if context fails
        alert("Web Audio API not supported. Audio feedback will be disabled.");
    }


    if (config.showDebugInfo) {
        debugElement.style.display = 'block';
    }

    statusElement.textContent = "Status: Ready. Close your eyes to start.";
    detectFace(); // Start the detection loop
}

main();
`````` <file-path>
config.js

import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as tf from '@tensorflow/tfjs';
// We don't directly import @mediapipe/face_mesh, tfjs runtime handles it via solutionPath
import { config } from './config.js';

// --- DOM Elements ---
const video = document.getElementById('webcam');
const overlayCanvas = document.getElementById('overlay');
const overlayCtx = overlayCanvas.getContext('2d');
const statusElement = document.getElementById('status');
const instructionsElement = document.getElementById('instructions');

// Debug Elements
const debugElement = document.getElementById('debug');
const faceDetectedElement = document.getElementById('face-detected');
const eyesOpenElement = document.getElementById('eyes-open');
const avgEarElement = document.getElementById('avg-ear');
const headRotationYElement = document.getElementById('head-rotation-y');
const headRotationXElement = document.getElementById('head-rotation-x');
const gameStateElement = document.getElementById('game-state');
const lastDirectionElement = document.getElementById('last-direction');

// --- State Variables ---
let model, detector;
let audioContext;
let isGameRunning = false;
let lastDirection = null; // 'left', 'right', 'up', 'down', or null
let detectionFrameId; // To control the animation loop

// --- Audio Setup ---
const setupAudio = () => {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        // Resume context on user interaction if needed (browsers require this)
        const resumeAudio = async () => {
            if (audioContext && audioContext.state === 'suspended') {
                await audioContext.resume();
                console.log("AudioContext resumed.");
            }
            // Remove listeners after first interaction
            document.body.removeEventListener('click', resumeAudio, true);
            document.body.removeEventListener('touchstart', resumeAudio, true);
            document.body.removeEventListener('keydown', resumeAudio, true);
        };
        document.body.addEventListener('click', resumeAudio, { once: true, capture: true });
        document.body.addEventListener('touchstart', resumeAudio, { once: true, capture: true });
        document.body.addEventListener('keydown', resumeAudio, { once: true, capture: true });

    } catch (e) {
        console.error("Web Audio API is not supported or failed to initialize:", e);
        config.audioEnabled = false; // Disable audio if context fails
        updateStatus("Audio disabled: Web Audio API not supported.", "error");
    }
};

// Function to play a specific pitch
const playPitch = (frequency) => {
    if (!config.audioEnabled || !audioContext || audioContext.state !== 'running') return;

    try {
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.type = 'sine'; // sine, square, sawtooth, triangle
        oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);

        // Simple envelope: fade in/out quickly
        gainNode.gain.setValueAtTime(0, audioContext.currentTime);
        gainNode.gain.linearRampToValueAtTime(config.audioVolume, audioContext.currentTime + config.audioDuration * 0.1);
        gainNode.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + config.audioDuration);

        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + config.audioDuration);
    } catch (error) {
        console.error("Error playing pitch:", error);
    }
};

// --- Debounce Utility ---
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Debounced audio functions for each direction
const playLeftTurnSound = debounce(() => { playPitch(config.leftTurnPitch); if(config.showDebugInfo) lastDirectionElement.textContent = 'Left'; }, config.audioDebounceTime);
const playRightTurnSound = debounce(() => { playPitch(config.rightTurnPitch); if(config.showDebugInfo) lastDirectionElement.textContent = 'Right'; }, config.audioDebounceTime);
const playLookUpSound = debounce(() => { playPitch(config.lookUpPitch); if(config.showDebugInfo) lastDirectionElement.textContent = 'Up'; }, config.audioDebounceTime);
const playLookDownSound = debounce(() => { playPitch(config.lookDownPitch); if(config.showDebugInfo) lastDirectionElement.textContent = 'Down'; }, config.audioDebounceTime);
const resetDirectionSound = debounce(() => { if(config.showDebugInfo) lastDirectionElement.textContent = 'Center'; }, config.audioDebounceTime);


// --- Face & Eye Analysis ---

// Landmark indices for eyes (adjust if model changes significantly)
// These are standard indices for MediaPipeFaceMesh
const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]; // Inner-outer, top-bottom points
const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]; // Inner-outer, top-bottom points

// Function to calculate Eye Aspect Ratio (EAR) - standard measure for eye openness
function calculateEAR(eyeLandmarks) {
    // Vertical distances (p2-p6, p3-p5)
    const v1 = Math.hypot(eyeLandmarks[1].x - eyeLandmarks[5].x, eyeLandmarks[1].y - eyeLandmarks[5].y);
    const v2 = Math.hypot(eyeLandmarks[2].x - eyeLandmarks[4].x, eyeLandmarks[2].y - eyeLandmarks[4].y);
    // Horizontal distance (p1-p4)
    const h = Math.hypot(eyeLandmarks[0].x - eyeLandmarks[3].x, eyeLandmarks[0].y - eyeLandmarks[3].y);

    if (h === 0) return 1.0; // Avoid division by zero, assume open

    const ear = (v1 + v2) / (2.0 * h);
    return ear;
}

// Simplified Head Pose Estimation (Yaw and Pitch)
function estimateHeadRotation(landmarks) {
    // Use key landmarks relative positions as proxy for rotation.
    // This is NOT a precise 3D pose estimation but sufficient for simple directional input.
    // Assumes landmarks are normalized (0.0 - 1.0) relative to video dimensions.
    const noseTip = landmarks[1];
    const leftEyeInner = landmarks[133]; // Left eye corner near nose
    const rightEyeInner = landmarks[362]; // Right eye corner near nose
    const chin = landmarks[152];
    const forehead = landmarks[10]; // A point approx on the forehead center

    // Yaw (Left/Right): Compare horizontal distance from nose to inner eye corners.
    // Larger distance between rightEyeInner.x and nose.x -> looking left (head turned right in mirrored view)
    // Larger distance between nose.x and leftEyeInner.x -> looking right (head turned left in mirrored view)
    const eyeCenterX = (leftEyeInner.x + rightEyeInner.x) / 2;
    // Adjusting calculation: positive yaw = head turned right (looking left in mirrored view)
    const yaw = noseTip.x - eyeCenterX;

    // Pitch (Up/Down): Compare vertical distance between forehead/nose and nose/chin.
    // Larger distance between forehead.y and noseTip.y -> looking down
    // Larger distance between noseTip.y and chin.y -> looking up
    // Use normalized y-coords (smaller y is higher up)
    const pitch = noseTip.y - (forehead.y + chin.y) / 2; // Center point between forehead/chin vs nose

    return { yaw, pitch };
}


// --- Game Logic ---
function startGame() {
    if (isGameRunning) return; // Don't restart if already running
    console.log("Game Started!");
    isGameRunning = true;
    updateStatus("Game Running! Move your head.", "running");
    instructionsElement.style.display = 'none'; // Hide instructions when game starts
    if(config.showDebugInfo) gameStateElement.textContent = 'Running';
    // --- ADD ANY GAME INITIALIZATION CODE HERE ---
}

function pauseGame(reason = "Paused") {
    if (!isGameRunning) return; // Don't pause if not running
    console.log(`Game Paused: ${reason}`);
    isGameRunning = false;
    updateStatus(`Paused (${reason}). Close eyes to resume.`, "warning");
    instructionsElement.style.display = 'block'; // Show instructions when paused
    lastDirection = null; // Reset direction on pause
    if(config.showDebugInfo) {
        gameStateElement.textContent = 'Paused';
        lastDirectionElement.textContent = 'None';
    }
    // --- ADD ANY GAME PAUSE CODE HERE ---
}

// --- Main Detection Loop ---
async function detectFace() {
    if (!detector || !video || video.readyState < HTMLMediaElement.HAVE_METADATA) {
        detectionFrameId = requestAnimationFrame(detectFace); // Keep trying
        return;
    }

    let faces;
    try {
        faces = await detector.estimateFaces(video, {
            flipHorizontal: false, // Input is already flipped by CSS/video transform
        });
    } catch (error) {
        console.error("Error during face estimation:", error);
        updateStatus("Face detection error.", "error");
        detectionFrameId = requestAnimationFrame(detectFace); // Try again
        return;
    }


    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height); // Clear previous drawings

    if (faces && faces.length > 0) {
        const face = faces[0];
        // Ensure keypoints exist and have expected structure
        if (!face.keypoints || face.keypoints.length === 0) {
             console.warn("Face detected, but no keypoints found.");
             if (isGameRunning) pauseGame("Tracking lost");
             updateStatus("Cannot track features. Adjust position?", "warning");
             if(config.showDebugInfo) faceDetectedElement.textContent = 'Yes (No Keypoints)';
             detectionFrameId = requestAnimationFrame(detectFace);
             return; // Skip further processing for this frame
        }

        const landmarks = face.keypoints;
        if(config.showDebugInfo) faceDetectedElement.textContent = 'Yes';


        // 1. Eye Tracking
        let leftEAR, rightEAR, avgEAR;
        let eyesAreOpen;
        try {
            const leftEyeLandmarks = LEFT_EYE_INDICES.map(i => landmarks[i]);
            const rightEyeLandmarks = RIGHT_EYE_INDICES.map(i => landmarks[i]);
            leftEAR = calculateEAR(leftEyeLandmarks);
            rightEAR = calculateEAR(rightEyeLandmarks);
            avgEAR = (leftEAR + rightEAR) / 2.0;
            eyesAreOpen = avgEAR > config.eyeClosedThreshold;

            if(config.showDebugInfo) {
                eyesOpenElement.textContent = eyesAreOpen ? 'True' : 'False';
                avgEarElement.textContent = avgEAR.toFixed(3);
            }

            // Draw Eye Landmarks if debugging
            if (config.drawLandmarks) {
                 overlayCtx.fillStyle = eyesAreOpen ? 'aqua' : 'magenta';
                 [...leftEyeLandmarks, ...rightEyeLandmarks].forEach(p => {
                     overlayCtx.beginPath();
                     // Map normalized coords to canvas coords
                     overlayCtx.arc(p.x * overlayCanvas.width, p.y * overlayCanvas.height, 2, 0, 2 * Math.PI);
                     overlayCtx.fill();
                 });
            }

        } catch (e) {
             console.error("Error calculating EAR:", e, "Landmarks:", landmarks.length);
             // Handle potential errors if landmarks are missing (should be caught earlier, but belt-and-suspenders)
             eyesAreOpen = true; // Assume eyes are open if calculation fails
             if(config.showDebugInfo) eyesOpenElement.textContent = 'Error';
        }


        // 2. Head Pose Estimation (if game is running)
        let rotation = { yaw: 0, pitch: 0 };
        if (isGameRunning) {
             try {
                rotation = estimateHeadRotation(landmarks);
                if(config.showDebugInfo) {
                    headRotationYElement.textContent = rotation.yaw.toFixed(3);
                    headRotationXElement.textContent = rotation.pitch.toFixed(3);
                }
             } catch (e) {
                 console.error("Error estimating head rotation:", e);
                 if(config.showDebugInfo) {
                    headRotationYElement.textContent = 'Error';
                    headRotationXElement.textContent = 'Error';
                }
             }
        } else {
             if(config.showDebugInfo) {
                 headRotationYElement.textContent = '...';
                 headRotationXElement.textContent = '...';
             }
        }


        // 3. Game State Logic
        if (!eyesAreOpen) {
            // Eyes are Closed - Start or Continue Game
            if (!isGameRunning) {
                startGame();
                // Recalculate rotation on the first frame the game starts
                 try {
                    rotation = estimateHeadRotation(landmarks);
                 } catch (e) { console.error("Error estimating rotation on game start:", e); }
            }

            // Process Head Movement for Audio Cues (only if game is running)
            let currentDirection = null;
            if (rotation.yaw > config.headRotationThresholdY) { // Turned head right (looking left in mirror)
                currentDirection = 'left'; // Sound for turning head right
                if (lastDirection !== currentDirection) playLeftTurnSound();
            } else if (rotation.yaw < -config.headRotationThresholdY) { // Turned head left (looking right in mirror)
                currentDirection = 'right'; // Sound for turning head left
                if (lastDirection !== currentDirection) playRightTurnSound();
            } else if (rotation.pitch > config.headRotationThresholdX) { // Looking down
                currentDirection = 'down';
                 if (lastDirection !== currentDirection) playLookDownSound();
            } else if (rotation.pitch < -config.headRotationThresholdX) { // Looking up
                 currentDirection = 'up';
                 if (lastDirection !== currentDirection) playLookUpSound();
            } else {
                // Head is relatively centered
                currentDirection = null;
                 if (lastDirection !== null) { // Only reset if previously moving
                    resetDirectionSound(); // Optional: Play a centering sound or just reset state
                }
            }
            lastDirection = currentDirection; // Update last detected direction


        } else {
            // Eyes are Open - Pause Game
            if (isGameRunning) {
                pauseGame("Eyes opened");
            } else {
                 // Ensure status is correct if paused for other reasons before
                 if (!statusElement.textContent.startsWith("Paused")) {
                    updateStatus("Ready. Close eyes to start.", "ready");
                 }
            }
        }

        // Draw All Landmarks if debugging (after potential eye drawing)
        if (config.drawLandmarks) {
            overlayCtx.fillStyle = 'lime'; // General landmark color
            landmarks.forEach((point, index) => {
                // Skip eye points if already drawn differently
                if (![...LEFT_EYE_INDICES, ...RIGHT_EYE_INDICES].includes(index)) {
                     overlayCtx.beginPath();
                     overlayCtx.arc(point.x * overlayCanvas.width, point.y * overlayCanvas.height, 1.5, 0, 2 * Math.PI);
                     overlayCtx.fill();
                }
            });
        }

    } else {
        // No face detected
        if(config.showDebugInfo) faceDetectedElement.textContent = 'No';
        if(config.showDebugInfo) eyesOpenElement.textContent = '...';
        if(config.showDebugInfo) avgEarElement.textContent = '...';
        if (isGameRunning) {
            pauseGame("Face lost");
        } else {
             updateStatus("No face detected. Please center yourself in the camera.", "warning");
        }
        lastDirection = null; // Reset direction
         if(config.showDebugInfo) lastDirectionElement.textContent = 'None';
    }

    // Loop the detection
    detectionFrameId = requestAnimationFrame(detectFace);
}

// --- Initialization ---
async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user', // Prefer front camera
                width: { ideal: 640 }, // Request a reasonable resolution
                height: { ideal: 480 }
            },
            audio: false
        });
        video.srcObject = stream;

        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();

                // Set canvas size based on video aspect ratio to fill screen (letterboxed)
                const videoAspect = video.videoWidth / video.videoHeight;
                const windowAspect = window.innerWidth / window.innerHeight;

                if (windowAspect > videoAspect) { // Window is wider than video
                    overlayCanvas.height = window.innerHeight;
                    overlayCanvas.width = window.innerHeight * videoAspect;
                } else { // Window is taller than video
                    overlayCanvas.width = window.innerWidth;
                    overlayCanvas.height = window.innerWidth / videoAspect;
                }
                // Center the canvas
                overlayCanvas.style.left = `${(window.innerWidth - overlayCanvas.width) / 2}px`;
                overlayCanvas.style.top = `${(window.innerHeight - overlayCanvas.height) / 2}px`;

                console.log(`Webcam initialized: ${video.videoWidth}x${video.videoHeight}. Canvas: ${overlayCanvas.width}x${overlayCanvas.height}`);
                resolve();
            };
            video.onerror = (e) => {
                 console.error("Video error:", e);
                 updateStatus("Webcam error.", "error");
                 resolve(); // Resolve anyway so the rest of the setup can proceed/fail gracefully
            }
        });
    } catch (err) {
        console.error("Error accessing webcam:", err);
        let message = "Webcam access denied or unavailable.";
        if (err.name === "NotAllowedError") {
             message = "Webcam permission denied. Please grant access and refresh.";
        } else if (err.name === "NotFoundError") {
             message = "No webcam found. Please ensure a camera is connected.";
        }
        updateStatus(message, "error");
        alert(message); // Also alert user
        return Promise.resolve(); // Resolve so main doesn't hang
    }
}

async function loadModel() {
    updateStatus("Loading face detection model...", "loading");
    try {
        // Ensure TFJS backend is set BEFORE creating the detector
        await tf.setBackend('webgl'); // Or 'wasm' or 'cpu'
        await tf.ready(); // Wait for backend to be ready
        console.log(`Using TensorFlow.js backend: ${tf.getBackend()}`);

        model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        detector = await faceLandmarksDetection.createDetector(model, config.modelConfig);
        console.log("Face detection model loaded successfully.");
        updateStatus("Model loaded. Setting up webcam...", "loading");
    } catch (error) {
        console.error("Error loading the model or setting backend:", error);
        updateStatus("Error loading face detection model.", "error");
        alert("Failed to load the face detection model. Check console for details (WebGL support might be required).");
        throw error; // Prevent further execution if model fails
    }
}

// --- Helper to update status message and style ---
function updateStatus(message, type = "info") { // types: info, warning, error, ready, loading, running
    statusElement.textContent = `Status: ${message}`;
    statusElement.className = type; // Assign class for styling
}


// --- Main Function ---
async function main() {
    console.log("Initializing application...");
    if (config.showDebugInfo) {
        debugElement.style.display = 'block';
    }

    try {
        await loadModel();
        setupAudio(); // Setup audio context early, needs interaction to resume
        await setupWebcam();

        if (!video.srcObject) {
            // Webcam setup failed, status already updated by setupWebcam
            console.error("Webcam stream not available. Stopping initialization.");
            return;
        }

        updateStatus("Ready. Close eyes to start.", "ready");
        detectionFrameId = requestAnimationFrame(detectFace); // Start the detection loop

    } catch (error) {
        // Errors during model load are already handled, this catches other potential issues
        console.error("Initialization failed:", error);
        updateStatus("Initialization failed. Please refresh.", "error");
    }

    // Handle window resize to keep canvas centered/sized correctly
    window.addEventListener('resize', () => {
         if(video.readyState >= HTMLMediaElement.HAVE_METADATA) {
             const videoAspect = video.videoWidth / video.videoHeight;
             const windowAspect = window.innerWidth / window.innerHeight;

             if (windowAspect > videoAspect) {
                 overlayCanvas.height = window.innerHeight;
                 overlayCanvas.width = window.innerHeight * videoAspect;
             } else {
                 overlayCanvas.width = window.innerWidth;
                 overlayCanvas.height = window.innerWidth / videoAspect;
             }
             overlayCanvas.style.left = `${(window.innerWidth - overlayCanvas.width) / 2}px`;
             overlayCanvas.style.top = `${(window.innerHeight - overlayCanvas.height) / 2}px`;
             console.log("Resized canvas:", overlayCanvas.width, overlayCanvas.height);
         }
    });
}

// --- Run ---
main();

