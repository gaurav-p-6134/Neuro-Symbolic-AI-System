# 🚧 Neuro-Symbolic Real-Time Hazard Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-18-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Gemini](https://img.shields.io/badge/AI-Gemini%201.5%20Flash-orange)
![YOLOv8](https://img.shields.io/badge/Computer%20Vision-YOLOv8-purple)

## 📋 Overview

This project implements a **Neuro-Symbolic AI architecture** for industrial safety monitoring. It combines the speed of real-time object detection (YOLOv8) with the reasoning capabilities of Multimodal Large Language Models (Google Gemini 1.5) and the statistical precision of Gradient Boosting (LightGBM).

Unlike standard "black box" AI, this system uses a hybrid approach:
1.  **Perception:** YOLOv8 detects objects in real-time.
2.  **Reasoning:** Gemini 1.5 analyzes the scene context (Weather, Road Type, Lighting) upon trigger.
3.  **Risk Calibration:** A pre-trained LightGBM classifier (trained on 55k+ Kaggle records) calculates a precise "Lethality Risk Score" based on the extracted features.

## 🚀 Key Features

* **Hybrid AI Pipeline:** seamlessly integrates Computer Vision, Generative AI, and Structured ML.
* **Real-Time Dashboard:** React + Vite frontend displaying live video feed and sub-200ms AI inference results.
* **Feature Extraction Agent:** Uses Gemini 1.5 Flash to convert unstructured video data into structured tabular data (JSON) compatible with classical ML models.
* **Legacy Model Integration:** Reuses a high-performance LightGBM model (Top 2% Global Rank) for risk scoring.

## 🏗️ Architecture

```mermaid
graph TD
    A[Webcam Feed] -->|Stream| B(FastAPI Backend)
    B -->|Frame| C{YOLOv8 Detection}
    C -->|BBox Overlay| D[React Frontend]
    D -->|User Triggers Scan| E[Gemini 1.5 Vision]
    E -->|Extract Features| F(JSON: Weather, Road, Light)
    F -->|Map to Schema| G[LightGBM Classifier]
    G -->|Risk Probability| D
