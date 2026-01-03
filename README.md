# Gesture-Controlled Competitive Painting Game 🎨✋

A real-time interactive multiplayer painting game built using **Computer Vision**.  
Players use **hand gestures detected via webcam** to paint inside a shared canvas. Each hand is assigned a color, and the winner is determined based on pixel coverage within a defined region.

---

## 🔥 Overview

This project demonstrates a complete real-time computer vision pipeline:
- Live webcam feed processing
- Hand landmark detection
- Gesture-based interaction
- Competitive gameplay mechanics
- Pixel-based scoring

Two players compete by painting inside a fixed **Region of Interest (ROI)** using their **thumb position**, detected via MediaPipe.

---

## 🧠 Key Features

- Real-time hand tracking using **MediaPipe Hands**
- Multi-hand detection with **left vs right hand differentiation**
- Gesture-controlled painting using thumb movement
- Alpha-blended custom brush rendering
- Countdown timer + fixed-duration gameplay
- Pixel-accurate scoring system
- Automatic winner declaration
- Dynamic canvas reset for replayability

---

## 🛠 Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe**
- **NumPy**

---

## 🎮 How It Works

1. Webcam captures a live video stream
2. MediaPipe detects hand landmarks in real time
3. Thumb tip position controls brush painting
4. Left and right hands are assigned different colors
5. Painting is restricted to a defined ROI
6. Game runs for a fixed duration
7. Pixel coverage inside the ROI determines the winner

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/gesture-painting-game.git
cd gesture-painting-game
```

### 2. Install dependencies
```bash
pip install opencv-python mediapipe numpy
```

### 3. Run the project
```bash
python main.py
```

> Press **`s`** to start the game  
> Press **`q`** to quit

---

## 📁 Project Structure

```
.
├── main.py
├── images/
│   ├── brush_1.png
│   ├── brush_2.png
│   └── image.webp
└── README.md
```

---

## 🚧 Future Improvements

- Gesture-based brush size control
- Gesture to clear canvas
- Sound effects and visual feedback
- Multiplayer score history
- UI enhancements

---

## 💡 Learning Outcomes

- Real-time computer vision pipelines
- Hand landmark processing
- Alpha blending and image overlays
- Game state management
- Pixel-level image analysis
- Interactive system design

---

## 📜 License

This project is for educational and portfolio purposes.
