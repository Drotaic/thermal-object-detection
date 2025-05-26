# Thermal Object Detection

Object detection system using TensorFlow to identify humans or heat anomalies in thermal imagery, captured by drones during disaster response missions.

## 🔥 Use Case
Thermal cameras on UAVs help detect survivors, fires, or animals hidden from standard vision. This project trains a real-time object detector on thermal data.

## 🧠 Model
- TensorFlow Object Detection API
- SSD or YOLO model fine-tuned on thermal images
- Outputs bounding boxes + confidence scores

## 📁 Structure
- `notebooks/`: Training pipeline & evaluation
- `scripts/`: Detection logic for images/video
- `models/`: Exported trained models
- `assets/`: Sample visual results
- `data/`: Thermal training dataset (images + labels)

## 🛠️ How to Use
1. Add labeled dataset to `data/`
2. Train model using `thermal_detection_notebook.ipynb`
3. Run detection using `scripts/thermal_detect.py`

## 🔮 Future Work
- Real-time drone feed integration
- Multi-class heat signature detection (people, animals, fire)
