# Predicting Diabetic Retinopathy Progression with Time-Series Databases and Deep Learning

### Overview
This repository contains work in progress on **predicting diabetic retinopathy progression** using segmentation and model blending techniques. The project is moving toward a paper tentatively titled:

**“Predicting Diabetic Retinopathy Progression with Time-Series Databases and Deep Learning.”**
---
#### Our current Kaggle Notebook being used


Notebook: [Link](https://www.kaggle.com/code/sunfish141/unet-for-diabetic-retinopathy)

---

### Teams

#### Segmentation Team
- Focused on analyzing medical images to isolate important regions such as lesions, optic discs, and other retinal structures.
- This step improves the precision of downstream models by providing clean, labeled inputs for training and evaluation.



#### Model Blending Team
- Involves integrating the predictions of multiple deep learning architectures to enhance overall accuracy and robustness.
- By combining results from different models and datasets, the blended approach aims to achieve more reliable progression prediction and classification outcomes.

---

### Datasets
- **DeepDRiD** – [GitHub](https://github.com/deepdrdoc/DeepDRiD)  
- **TJDR** – Pixel-level lesion annotations for 561 fundus images.  
- **APTOS 2019 Blindness Detection** – [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

---

### Relevant Papers:
- "Towards Predicting Temporal Changes in a Patient's Condition from Image Sequences" [link](arXiv:2409.07012)
- "DRetNet: A Novel Deep Learning Framework for Diabetic Retinopathy Diagnosis" [link](arXiv:2509.01072)

---

### References
- [Nature Scientific Data 2025 – Fundus Image Segmentation](https://www.nature.com/articles/s41597-025-04627-3)  
- [APTOS 2019 1st Place Solution Summary](https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary)
```
