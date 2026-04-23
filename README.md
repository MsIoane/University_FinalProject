# Evaluating and Mitigating Bias in AI-Based Credit Scoring Systems

Bachelor's Research Thesis — Computer Science, Georgian Technical University, 2026

## Overview

This repository contains the thesis document and Python implementation
for a study on demographic bias in AI-driven credit scoring systems.
The study detects, measures, and mitigates bias across three protected
attributes — sex, age, and foreign worker status — using the German
Credit Dataset, and evaluates the fairness-accuracy tradeoff of a
Reweighting pre-processing intervention.

## Key Results

| Metric | Before Mitigation | After Mitigation |
|---|---|---|
| Gender SPD | -0.416 | -0.094 ✅ |
| Gender DIR | 0.568 | 0.851 ✅ |
| Model Accuracy | 60.0% | 56.0% |
| SPD Improvement | — | 77.4% |

## Repository Structure
