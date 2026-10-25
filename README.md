# CAST-CD

**Remote Sensing Image Change Detection · Final-Year Project**

CAST-CD is a final-year Computer Science and Engineering project at **Amrita
Vishwa Vidyapeetham, Coimbatore**, investigating deep learning for change
detection in remote sensing imagery.

Given two images of the same geographical region captured at different times,
the project aims to identify areas that have changed and produce a pixel-level
binary change map. The research focuses on balancing detection quality with
computational efficiency.

## Project Overview

Remote sensing imagery helps reveal how the Earth's surface changes over time.
Detecting meaningful changes is challenging because lighting, seasonal
variation, and image noise can also alter the appearance of a scene.

This project explores automated change detection with potential applications in:

- Urban growth and infrastructure monitoring.
- Land-cover and environmental change analysis.
- Disaster impact assessment.

These are motivating applications; the project is an academic research prototype.

## Objectives

- Investigate efficient deep learning approaches for paired remote sensing images.
- Identify meaningful changes while reducing sensitivity to appearance variations.
- Assess detection quality and computational requirements through experiments.
- Document the research as part of the final-year project and a planned paper.

## Research Status

**Active research · Manuscript in preparation**

This README provides a public overview of the project. Detailed methodology,
architecture specifications, training configurations, experimental results,
and ablation studies are intentionally omitted here while the manuscript is
being prepared.

The repository is a development workspace. Its contents may evolve and should
not be treated as a finalized release accompanying a published paper. Publication
details and a formal citation will be added when available.

## Repository Organization

| Location | Purpose |
| --- | --- |
| `cast-cd/` | CAST-CD research implementation and development resources. |
| `other_models/` | Additional model resources for comparative experimentation. |

## Technology

The implementation uses **Python** and **PyTorch**, with Torchvision, NumPy,
OpenCV, Pillow, and tqdm supporting image processing and experimentation.

## Academic Team

Aditya Suresh · Ananthasagar N · Pratyush Yadav · Karthik P · Suriakiran B ·
Jayakrishnan Anandakrishnan

**Department of Computer Science and Engineering**

Amrita Vishwa Vidyapeetham, Coimbatore, Tamil Nadu, India

## Acknowledgements

We acknowledge the researchers and maintainers of the datasets, frameworks,
and existing models used in this project. Third-party resources retain their
original attribution and applicable terms; consult their accompanying
documentation for details.
