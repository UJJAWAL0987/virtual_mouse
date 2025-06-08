# Virtual Mouse Using Hand Gestures

A Python-based desktop application that allows users to control their mouse cursor using hand gestures captured through a webcam.

## Features

- **Cursor Movement**: Control mouse cursor using index finger
- **Left Click**: Touch thumb and index finger together
- **Right Click**: Form a circle with thumb, index, and middle fingers
- **Scroll**: Swipe up/down with index finger
- **Smooth Movement**: Cursor movement is smoothed using moving average
- **Real-time FPS Display**: Shows current frames per second

## Requirements

- Python 3.x
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd virtual-mouse
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

## Usage

1. Run the application:
```bash
python main.py
```

2. Position yourself in front of the webcam with good lighting.

3. Use the following gestures:
   - Move your index finger to control the cursor
   - Touch thumb and index finger together for left click
   - Form a circle with thumb, index, and middle fingers for right click
   - Swipe up/down with index finger to scroll

4. Press 'q' to quit the application.

## Gesture Guide

1. **Cursor Movement**
   - Extend your index finger
   - Move your hand to control the cursor
   - Keep other fingers down for better control

2. **Left Click**
   - Touch your thumb and index finger together
   - Hold briefly for click
   - Release to prepare for next click

3. **Right Click**
   - Form a circle with thumb, index, and middle fingers
   - Hold briefly for right-click
   - Release to prepare for next action

4. **Scrolling**
   - Extend your index finger
   - Move up or down to scroll
   - Faster movement = faster scrolling

## Troubleshooting

1. **Poor Hand Detection**
   - Ensure good lighting
   - Keep your hand within the camera frame
   - Avoid rapid movements

2. **Cursor Jitter**
   - Keep your hand steady
   - Ensure good lighting
   - Adjust smoothing factor in code if needed

3. **Click Not Working**
   - Make sure fingers are clearly touching
   - Check lighting conditions
   - Try adjusting gesture thresholds in code

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 