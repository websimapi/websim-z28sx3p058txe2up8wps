export const config = {
    // Face detection model parameters
    modelConfig: {
        runtime: 'mediapipe', // or 'tfjs'
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/', // Only needed for mediapipe runtime
        maxFaces: 1,
        refineLandmarks: true, // Required for iris tracking
    },

    // Eye tracking thresholds
    eyeClosedThreshold: 0.3, // Lower value means more sensitivity to closing

    // Head rotation thresholds (in radians)
    headRotationThresholdY: 0.2, // Left/Right turn sensitivity
    headRotationThresholdX: 0.15, // Up/Down look sensitivity

    // Audio settings
    audioEnabled: true,
    leftTurnPitch: 300,
    rightTurnPitch: 500,
    lookUpPitch: 700,
    lookDownPitch: 200,
    audioDuration: 0.1, // seconds
    audioVolume: 0.5, // 0 to 1

    // Debug settings
    showDebugInfo: false, // Set to true to show debug overlay
    drawLandmarks: false, // Set to true to draw face landmarks on the canvas
};

export const config = {
    // Face detection model parameters
    modelConfig: {
        runtime: 'mediapipe', // or 'tfjs'
        // Using the full path to the mediapipe solution assets
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/', // Ensure this path is correct
        maxFaces: 1,
        refineLandmarks: true, // Required for iris tracking / better accuracy
    },

    // Eye tracking thresholds
    // Adjust based on sensitivity. Lower value means the eye needs to be more closed to trigger 'closed'.
    // Values typically range from 0.1 (very closed) to 0.4 (slightly closed).
    eyeClosedThreshold: 0.25, // Default: 0.25 - Requires significant closure

    // Head rotation thresholds (approximate radians derived from normalized coords)
    // Adjust these to change how much head movement is needed. Larger values require more movement.
    headRotationThresholdY: 0.1, // Left/Right turn sensitivity (Yaw) - Default: 0.1
    headRotationThresholdX: 0.08, // Up/Down look sensitivity (Pitch) - Default: 0.08

    // Audio settings
    audioEnabled: true,
    // Frequencies in Hz for different directions
    leftTurnPitch: 300,
    rightTurnPitch: 500,
    lookUpPitch: 700,
    lookDownPitch: 200,
    audioDuration: 0.1, // Duration of the sound in seconds
    audioVolume: 0.3, // Volume (0.0 to 1.0)

    // Debounce time for audio cues (milliseconds) to prevent rapid firing
    audioDebounceTime: 150, // ms

    // Debug settings
    showDebugInfo: false, // Set to true to show text debug info on screen
    drawLandmarks: false, // Set to true to draw face landmarks on the canvas overlay
};

