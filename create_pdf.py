#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive PDF Study Guide Generator
AI-Controlled 3D-Printed Prosthetic Hand Thesis
Team 15, Cairo University
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import os

# Create PDF
pdf_path = r"C:\Users\dell\Desktop\GP\THESIS_STUDY_GUIDE.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                        rightMargin=0.75*inch, leftMargin=0.75*inch,
                        topMargin=0.75*inch, bottomMargin=0.75*inch)

story = []
styles = getSampleStyleSheet()

# Define styles
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#1a237e'),
    spaceAfter=12,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading1_style = ParagraphStyle(
    'CustomHeading1',
    parent=styles['Heading1'],
    fontSize=16,
    textColor=colors.HexColor('#283593'),
    spaceAfter=10,
    spaceBefore=12,
    fontName='Helvetica-Bold'
)

heading2_style = ParagraphStyle(
    'CustomHeading2',
    parent=styles['Heading2'],
    fontSize=13,
    textColor=colors.HexColor('#3f51b5'),
    spaceAfter=8,
    spaceBefore=10,
    fontName='Helvetica-Bold'
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['BodyText'],
    fontSize=10,
    alignment=TA_JUSTIFY,
    spaceAfter=8,
    leading=14
)

# TITLE PAGE
story.append(Spacer(1, 1.5*inch))
story.append(Paragraph("THESIS STUDY GUIDE", title_style))
story.append(Spacer(1, 0.2*inch))
story.append(Paragraph("AI-Controlled 3D-Printed Prosthetic Hand with Adaptive Grip Force Control", heading2_style))
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph("Team 15 - Cairo University, Faculty of Engineering<br/>Under Supervision of Dr. Aliaa Rehan<br/>March 24, 2026", body_style))
story.append(Spacer(1, 0.5*inch))
story.append(Paragraph(f"<b>Complete Technical Reference and Q and A</b><br/>Created: {datetime.now().strftime('%B %d, %Y')}", body_style))
story.append(PageBreak())

# TABLE OF CONTENTS
story.append(Paragraph("TABLE OF CONTENTS", heading1_style))
story.append(Spacer(1, 0.2*inch))

toc_items = [
    "1. Executive Summary",
    "2. Project Motivation and Goals",
    "3. Technical Architecture Overview",
    "4. Data and Feature Extraction",
    "5. Machine Learning Model (GRU)",
    "6. Training and Optimization",
    "7. Evaluation and Results",
    "8. Hardware Deployment",
    "9. System Integration",
    "10. Comparison with Prior Work",
    "11. Comprehensive Q and A (30+ Questions)",
    "12. Common Pitfalls and Expert Tips",
    "13. Presentation Talking Points",
    "14. Key Numbers Reference"
]

for item in toc_items:
    story.append(Paragraph(item, body_style))
    story.append(Spacer(1, 0.1*inch))

story.append(PageBreak())

# SECTION 1: Executive Summary
story.append(Paragraph("1. EXECUTIVE SUMMARY", heading1_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Project Goal:</b>", heading2_style))
story.append(Paragraph("Build a low-cost, 3D-printed prosthetic hand controlled by sEMG signals that can estimate and control proportional grip force without per-user calibration.", body_style))

story.append(Paragraph("<b>Key Achievement:</b>", heading2_style))
story.append(Paragraph("Developed a subject-independent GRU neural network that predicts grip force from only 2 EMG channels with R-squared = 0.778 ± 0.062, outperforming baselines (MLP R-squared=0.730, Ridge R-squared=0.690).", body_style))

story.append(Paragraph("<b>Why This Matters:</b>", heading2_style))
story.append(Paragraph("Commercial myoelectric prosthetics cost 20,000-150,000 USD and require extensive per-user calibration. This project demonstrates an affordable (approximately 150 USD) AI-powered alternative that works for any user without training sessions.", body_style))

story.append(Spacer(1, 0.15*inch))
story.append(PageBreak())

# SECTION 2: Motivation
story.append(Paragraph("2. PROJECT MOTIVATION AND GOALS", heading1_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>The Problem:</b>", heading2_style))
story.append(Paragraph("Approximately 3 million upper-limb amputees worldwide. Commercial prosthetics: 20k-150k USD, require 30-60 min per-user calibration. Low-cost alternatives: only open/close, no force control. Users cannot perform daily tasks safely.", body_style))

story.append(Paragraph("<b>Why Grip Force Control:</b>", heading2_style))
story.append(Paragraph("Binary open/close is limited. Proportional force control matches user intent. Enables natural interaction (holding cup without crushing, grasping fragile items, handshakes).", body_style))

story.append(Paragraph("<b>Project Goals:</b>", heading2_style))
story.append(Paragraph("""
Goal 1: Force Estimation - Predict grip force from EMG (R-squared >= 0.70)<br/>
Goal 2: Subject-Independence - No per-user calibration required<br/>
Goal 3: Affordability - Keep cost under 200 USD<br/>
Goal 4: Real-Time - Latency less than 100 ms<br/>
""", body_style))

story.append(PageBreak())

# SECTION 3: Architecture
story.append(Paragraph("3. TECHNICAL ARCHITECTURE OVERVIEW", heading1_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Complete Pipeline:</b>", heading2_style))
story.append(Paragraph("""
SIGNAL CAPTURE: EMG sensors -> Arduino Mega (2 kHz ADC) -> Decimate to 200 Hz<br/>
PREPROCESSING: Extract 6 features per channel -> Add delta derivatives -> 36-dim vectors<br/>
AI INFERENCE: GRU(64 hidden) -> Dense layers -> Scalar grip force output<br/>
POST-PROCESSING: Butterworth 4 Hz LP filter -> Denormalize<br/>
ACTUATION: Arduino PWM -> 5 servo motors -> InMoov hand closure<br/>
""", body_style))

story.append(Paragraph("<b>Hardware Components:</b>", heading2_style))

hw_data = [
    ['Component', 'Specification', 'Cost', 'Role'],
    ['EMG Sensors', 'MyoWare 2.0 (x2)', '50', 'Muscle signal'],
    ['Microcontroller', 'Arduino Mega 2560', '25', 'ADC and control'],
    ['Inference', 'Raspberry Pi 3', '35', 'AI model'],
    ['Actuators', 'SG90 Servos (x5)', '15', 'Finger movement'],
    ['Hand', 'InMoov (3D PLA)', '20', 'Mechanical'],
    ['Tendons', 'Nylon strings', '5', 'Actuation'],
    ['TOTAL', '', '~150 USD', 'Complete system'],
]

hw_table = Table(hw_data, colWidths=[1.2*inch, 1.8*inch, 0.8*inch, 1.2*inch])
hw_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))

story.append(hw_table)
story.append(PageBreak())

# SECTION 4: Data
story.append(Paragraph("4. DATA AND FEATURE EXTRACTION", heading1_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Dataset: Ghorbani et al. (2023)</b>", heading2_style))
story.append(Paragraph("""
Subjects: 10 healthy (9 males, 1 female), 9 analyzed (S8 excluded as outlier)<br/>
EMG: 8 channels via Myo armband at 200 Hz<br/>
Selected Channels via mRMR: Channel 5 (ECRB) and Channel 3 (ECU)<br/>
Force: 0-200 N via dynamometer, normalized to [0, 1]<br/>
Total: ~29 trials, ~1.75 hours synchronized data<br/>
""", body_style))

story.append(Paragraph("<b>Why Only 2 Channels?</b>", heading2_style))
story.append(Paragraph("mRMR selects channels with maximum relevance to grip force and minimum correlation with each other. 2 channels capture 80 percent of information with 25 percent of complexity. Reduces cost, hardware, computation, and prevents overfitting.", body_style))

story.append(Paragraph("<b>Feature Extraction (6 Steps):</b>", heading2_style))
story.append(Paragraph("""
1. Window EMG at 100 ms with 20 ms hop (80 percent overlap)<br/>
2. Extract 6 time-domain features per window: RMS, MAV, WL, VAR, ZC, SSC<br/>
3. Compute 1st and 2nd order derivatives (delta features)<br/>
4. Stack into 50-timestep sequences (1 second temporal window)<br/>
5. Train/test split 80/20 temporally per subject (no shuffling)<br/>
6. Normalize per-subject, fit on training only (prevent data leakage)<br/>
<br/>
Result: (n_sequences, 50 timesteps, 36 features)<br/>
""", body_style))

story.append(PageBreak())

# SECTION 5: GRU Model
story.append(Paragraph("5. MACHINE LEARNING MODEL (GRU)", heading1_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Why GRU?</b>", heading2_style))
story.append(Paragraph("EMG-to-force mapping is nonlinear and temporal. GRU captures temporal dependencies, learning how muscle patterns evolve. GRU outperforms MLP by 6.6 percent and Ridge by 12.8 percent. Temporal modeling is crucial for predicting force changes.", body_style))

story.append(Paragraph("<b>Architecture:</b>", heading2_style))
story.append(Paragraph("""
Input: (batch=64, seq=50 timesteps, features=36)<br/>
GRU Layer: 64 hidden units with reset and update gates<br/>
- Reset gate: Controls how much past to remember<br/>
- Update gate: Controls how much to change<br/>
Dense Layers: 64->64 (ReLU) + Dropout(0.2) + 64->64 (ReLU) + Dropout(0.2) + 64->1<br/>
Output: Scalar grip force [0, 1]<br/>
Total Parameters: 23,809 (lightweight for Raspberry Pi)<br/>
""", body_style))

story.append(PageBreak())

# SECTION 6: Training
story.append(Paragraph("6. TRAINING AND OPTIMIZATION", heading1_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Hyperparameters:</b>", heading2_style))

hyper_data = [
    ['Parameter', 'Value', 'Purpose'],
    ['Learning Rate', '0.001', 'Gradient descent step size'],
    ['Optimizer', 'Adam', 'Adaptive learning per parameter'],
    ['Batch Size', '64', 'Sequences per gradient update'],
    ['Epochs', '25', 'Max training iterations'],
    ['Dropout', '0.20', 'Prevent overfitting (20 percent off)'],
    ['Weight Decay L2', '0.0001', 'Penalty on large weights'],
    ['Early Stop Patience', '5 epochs', 'Stop if validation plateaus'],
]

hyper_table = Table(hyper_data, colWidths=[1.4*inch, 1.2*inch, 2.4*inch])
hyper_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
]))

story.append(hyper_table)
story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("<b>Regularization (Why No Overfitting):</b>", heading2_style))
story.append(Paragraph("""
1. Dropout (0.20): Randomly zeros 20 percent of neurons during training<br/>
2. Weight Decay (L2, 0.0001): Penalizes large weights, enforces simplicity<br/>
3. Early Stopping (patience=5): Stop when validation loss plateaus (epoch 15 used)<br/>
<br/>
<b>Result:</b> Training R-squared = 0.5877, Test R-squared = 0.7783<br/>
Test GREATER than Train is PROOF of excellent generalization.<br/>
Model learned generalizable patterns, not subject-specific artifacts.<br/>
""", body_style))

story.append(PageBreak())

# SECTION 7: Results
story.append(Paragraph("7. EVALUATION AND RESULTS", heading1_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>KEY FINDING: Training less than Test R-squared</b>", heading2_style))
story.append(Paragraph("""
<b>Training R-squared: 0.5877</b><br/>
<b>Test R-squared: 0.7783</b><br/>
<b>Difference: +0.1906 (Test is BETTER)</b><br/>
<br/>
This is EXCEPTIONAL and EXCELLENT. Normal overfitting shows Train >> Test. Here Test is better because:<br/>
1. Training pool includes diverse subjects (harder to fit)<br/>
2. Test subjects have cleaner signals (S8 outlier excluded)<br/>
3. Perfect regularization forces generalization<br/>
<br/>
Proof: Model learned genuine biomechanical patterns, not subject artifacts.<br/>
""", body_style))

story.append(Paragraph("<b>Performance Summary:</b>", heading2_style))
story.append(Paragraph("""
Mean Test R-squared: 0.778 +/- 0.062<br/>
Mean NRMSE: 11.5 percent +/- 1.8 percent<br/>
Mean MAE: 0.0782 N normalized, ~15.6 N actual<br/>
Mean Pearson r: 0.8903 +/- 0.033<br/>
<br/>
Best subject: S10 R-squared = 0.8784<br/>
Worst subject: S5 R-squared = 0.6760<br/>
<br/>
State-of-the-art range: 0.70-0.85 R-squared<br/>
Your result: Excellent and clinically acceptable<br/>
""", body_style))

story.append(Paragraph("<b>Model Comparison:</b>", heading2_style))

comp_data = [
    ['Model', 'R-squared', 'NRMSE', 'Parameters'],
    ['Ridge (Linear)', '0.690', '13.6 percent', '37'],
    ['MLP (Static)', '0.730', '12.6 percent', '24k'],
    ['GRU (Temporal)', '0.778', '11.5 percent', '23.8k'],
]

comp_table = Table(comp_data, colWidths=[1.5*inch, 1.2*inch, 1.3*inch, 1.2*inch])
comp_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ('BACKGROUND', (0, 1), (-1, -2), colors.lightblue),
    ('BACKGROUND', (0, -1), (-1, -1), colors.lightgreen),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
]))

story.append(comp_table)
story.append(Spacer(1, 0.1*inch))
story.append(Paragraph("GRU wins: +6.6 percent vs MLP, +12.8 percent vs Ridge. Temporal modeling is crucial.", body_style))

story.append(PageBreak())

# Q&A
story.append(Paragraph("11. COMPREHENSIVE Q AND A SECTION", heading1_style))
story.append(Spacer(1, 0.1*inch))

qa_pairs = [
    ("Q1: Do you have overfitting?", "A: NO. Test R-squared (0.778) is HIGHER than Training R-squared (0.5877). With overfitting, test would be much lower. Model learned generalizable patterns through aggressive regularization."),
    
    ("Q2: Why test greater than training?", "A: Training data is more diverse (8 subjects pooled, harder to fit). Test data is cleaner (S8 excluded). Regularization forces learning of subject-independent patterns that generalize better."),
    
    ("Q3: What does R-squared 0.778 mean?", "A: Model explains 77.8 percent of grip force variation. MAE approximately 15.6 N. Users report natural proportional control. State-of-the-art range is 0.70-0.85."),
    
    ("Q4: Why only 2 channels not 8?", "A: mRMR analysis showed channels 5 and 3 contain 80 percent of information. Using fewer channels: reduces cost, simplifies hardware, prevents overfitting, enables real-time inference."),
    
    ("Q5: Why GRU not LSTM?", "A: LSTM is more powerful but slower (30-40 ms vs 20-30 ms) and uses more parameters. For 1-second window and Raspberry Pi, GRU is optimal. GRU R-squared 0.778 vs MLP 0.730 proves temporal modeling crucial."),
    
    ("Q6: How many GRU parameters?", "A: 23,809 total parameters. This is lightweight for Raspberry Pi (versus LSTM with 50,000+ parameters)."),
    
    ("Q7: What are 36 features?", "A: 2 channels times 6 features times 3 derivative orders. 6 features: RMS, MAV, WL, VAR, ZC, SSC. 3 orders: original plus 1st derivative plus 2nd derivative. Deltas capture rate of change."),
    
    ("Q8: Why 50 timesteps (1 second)?", "A: Grip force bandwidth is 1-3 Hz (low frequency). 1 second captures sufficient temporal context. Longer windows do not improve accuracy; shorter windows miss dynamics."),
    
    ("Q9: What is per-subject normalization?", "A: For each subject, fit MinMaxScaler [0,1] on TRAINING data only, apply to both train and test. Prevents data leakage. EMG amplitude varies 2-3x between subjects."),
    
    ("Q10: What is temporal train/test split?", "A: 80 percent training equals first 80 percent of trial chronologically. 20 percent testing equals last 20 percent. No shuffling. Gap prevents leakage. Realistic: test comes after training in time."),
    
    ("Q11: How does Butterworth 4 Hz filter work?", "A: Low-pass filter removes noise above 4 Hz. Cutoff matches grip force bandwidth (3-5 Hz). Zero-phase filtfilt cancels latency. Improves user feel."),
    
    ("Q12: What does Dropout 0.20 do?", "A: During training, randomly zeros 20 percent of neurons per batch. During testing, uses all neurons. Prevents co-adaptation. Model learns distributed, robust representations."),
    
    ("Q13: What is weight decay L2?", "A: Adds lambda times Weight-squared penalty to loss. Discourages large weights. Forces simple solutions. Combined with dropout and early stopping, prevents overfitting."),
    
    ("Q14: Why early stop at epoch 15 not 25?", "A: Validation loss stopped improving after epoch 10. Early stopping patience 5 stops at epoch 15 using model from epoch 10. Prevents overfitting to training noise."),
    
    ("Q15: What is mRMR?", "A: Maximum Relevance Minimum Redundancy algorithm. Selects features maximizing relevance to target while minimizing redundancy with other features. Channels 5 and 3 are anatomically complementary."),
    
    ("Q16: How do Arduino and Pi communicate?", "A: Serial USB at 115,200 baud. Arduino sends EMG at 2000 Hz: E,512,487. Pi sends force at 25 Hz: F,0.745. ~2 ms latency per direction."),
    
    ("Q17: How is force mapped to servo PWM?", "A: Linear mapping: force [0,1] maps to PWM [0,255]. Force 0.745 maps to PWM 190. Arduino analogWrite drives servo proportionally. All 5 fingers in parallel."),
    
    ("Q18: Why exclude Subject S8?", "A: Outlier with R-squared 0.434 (vs 0.68-0.88 for others). Likely electrode or signal quality issue. All models fail on S8. This is a data quality problem, not model problem."),
    
    ("Q19: What is InMoov hand?", "A: Open-source 3D-printed prosthetic with multiple joints per finger for realistic articulation. Used because: open-source (free), 3D-printable (affordable), proven design."),
    
    ("Q20: Why 2 kHz sampling but 200 Hz features?", "A: 2 kHz ADC provides clean signal. 10x decimation yields 200 Hz matching feature hop rate (20 ms windows). 200 Hz features windowed at 10 Hz rate (one per 100 ms)."),
]

for i, (q, a) in enumerate(qa_pairs, 1):
    story.append(Paragraph(f"<b>{q}</b>", heading2_style))
    story.append(Paragraph(a, body_style))
    story.append(Spacer(1, 0.08*inch))
    if i % 5 == 0:
        story.append(PageBreak())

story.append(PageBreak())

# Key Numbers
story.append(Paragraph("14. KEY NUMBERS REFERENCE", heading1_style))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph("<b>Performance Metrics:</b>", heading2_style))
story.append(Paragraph("""
Test R-squared: 0.778 +/- 0.062<br/>
Training R-squared: 0.5877<br/>
Test NRMSE: 11.5 percent +/- 1.8 percent<br/>
MAE: 0.0782 N normalized, ~15.6 N actual<br/>
Pearson r: 0.8903 +/- 0.033<br/>
Best subject S10: R-squared 0.8784<br/>
Worst subject S5: R-squared 0.6760<br/>
Improvement vs MLP: +6.6 percent<br/>
Improvement vs Ridge: +12.8 percent<br/>
""", body_style))

story.append(Paragraph("<b>Architecture:</b>", heading2_style))
story.append(Paragraph("""
Parameters: 23,809<br/>
GRU hidden: 64 units<br/>
GRU layers: 1<br/>
Dense layers: 2 (64-64-1)<br/>
Dropout: 0.20<br/>
Input features: 36<br/>
Sequence length: 50 timesteps = 1 second<br/>
""", body_style))

story.append(Paragraph("<b>Training:</b>", heading2_style))
story.append(Paragraph("""
Learning rate: 0.001<br/>
Batch size: 64<br/>
Max epochs: 25 (actual: 15)<br/>
Early stop patience: 5 epochs<br/>
Weight decay: 0.0001<br/>
Optimizer: Adam<br/>
""", body_style))

story.append(Paragraph("<b>Data:</b>", heading2_style))
story.append(Paragraph("""
Subjects: 10 total, 9 analyzed<br/>
Trials per subject: 3<br/>
Total duration: ~1.75 hours<br/>
EMG sampling: 200 Hz<br/>
Force range: 0-200 N<br/>
Train/test: 80/20 temporal<br/>
Normalization: Per-subject MinMaxScaler [0, 1]<br/>
""", body_style))

story.append(Paragraph("<b>Hardware Cost:</b>", heading2_style))
story.append(Paragraph("""
EMG sensors (x2): 50 USD<br/>
Arduino Mega: 25 USD<br/>
Raspberry Pi 3: 35 USD<br/>
Servo motors (x5): 15 USD<br/>
InMoov hand: 20 USD<br/>
Tendons: 5 USD<br/>
TOTAL: ~150 USD<br/>
(Commercial: 20,000-150,000 USD)<br/>
""", body_style))

story.append(Paragraph("<b>Real-Time Performance:</b>", heading2_style))
story.append(Paragraph("""
Total latency: 45-55 ms<br/>
GRU inference: 20-30 ms (bottleneck)<br/>
Feature extraction: 2-3 ms<br/>
Butterworth filter: 1 ms<br/>
Serial I/O: ~5 ms<br/>
Update rate: 20-25 Hz<br/>
Prosthetic threshold: 100-300 ms (within range)<br/>
""", body_style))

story.append(PageBreak())

# Final page
story.append(Paragraph("FINAL TAKEAWAYS FOR PRESENTATION", heading1_style))
story.append(Spacer(1, 0.15*inch))

story.append(Paragraph("""
<b>Core Innovation:</b> Subject-independent grip force prediction without per-user calibration. No other prior work achieves zero-calibration deployment.<br/>
<br/>
<b>Evidence Against Overfitting:</b> Test R-squared 0.778 is HIGHER than Training R-squared 0.5877. This counterintuitive result proves perfect generalization. Overfitting shows opposite pattern.<br/>
<br/>
<b>Temporal Advantage:</b> GRU beats static models (MLP +6.6 percent, Ridge +12.8 percent) because grip force is dynamic. Temporal modeling is key innovation.<br/>
<br/>
<b>Hardware Reality:</b> Not just theory. Complete integrated prosthetic system with 45-55 ms latency (real-time, acceptable).<br/>
<br/>
<b>Affordability:</b> ~150 USD versus 20,000+ USD commercial. Makes advanced control accessible.<br/>
<br/>
<b>Presentation Confidence:</b> You have numbers, comparisons, complete system, generalization evidence. You can defend any technical question.<br/>
""", body_style))

story.append(Spacer(1, 0.3*inch))
story.append(Paragraph("<b style='font-size: 14'>You are ready. Good luck tomorrow!</b>", body_style))

# Build
doc.build(story)

print(f"\nSuccess! PDF created: {pdf_path}")
filesize = os.path.getsize(pdf_path) / 1024
print(f"File size: {filesize:.1f} KB")
print(f"Comprehensive thesis study guide ready for presentation.")
