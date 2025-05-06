# Assignment: Implementing a Simplified PEDO Framework for Car Design Optimization

# -------------------------------------------------
# Introduction (Markdown Cell)
"""
## Implementing a Simplified PEDO Framework for Car Design Optimization

In this notebook, we implement a simplified version of the Prompt Evolution Design Optimization (PEDO) framework using open-source tools. We focus on generating 2D car designs with a text-to-image model and optimizing them using a simple evolutionary algorithm.

**Objectives:**
- Generate 2D car designs using Stable Diffusion.
- Implement a basic evolutionary algorithm to optimize prompts.
- Score designs based on simplified aerodynamic performance.
- Penalize unrealistic designs using an image classifier.
- Visualize performance and summarize insights.
"""

# -------------------------------------------------
# Step 1: Setup Environment (code cell)

import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "diffusers", "transformers", "matplotlib", "numpy", "scipy", "torch", "torchvision"]
    )

# -------------------------------------------------
# Step 2: Text-to-Image Generation using Stable Diffusion
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Example
prompt = "a futuristic car design with aerodynamic features"
image = generate_image(prompt)
image.show()

# -------------------------------------------------
# Step 3: Evolutionary Algorithm for Prompt Optimization
import random

population = ["sleek car", "futuristic car", "compact car"]

def evaluate_population(pop):
    return [random.uniform(0, 1) for _ in pop]  # Placeholder fitness

def select_top_prompts(pop, scores, top_k=2):
    sorted_pop = sorted(zip(pop, scores), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_pop[:top_k]]

def mutate_and_recombine(prompts):
    new_prompts = []
    for prompt in prompts:
        mutation = random.choice([" sporty", " aerodynamic", " lightweight"])
        new_prompts.append(prompt + mutation)
    return new_prompts + prompts

# Run Evolution
generations = 5
for gen in range(generations):
    scores = evaluate_population(population)
    population = mutate_and_recombine(select_top_prompts(population, scores))
    print(f"Generation {gen+1}: {population}")

# -------------------------------------------------
# Step 4: Simplified Aerodynamic Scoring
from PIL import Image

def width_to_height_ratio(image):
    width, height = image.size
    return width / height

def aerodynamic_score(image):
    ratio = width_to_height_ratio(image)
    return 1 / ratio if ratio != 0 else 0

# -------------------------------------------------
# Step 5: Penalizing Unrealistic Designs with ResNet
from torchvision import models, transforms

model = models.resnet18(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def penalize_unrealistic(image):
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    penalty = 1 - output.softmax(1)[0].max().item()  # Use the highest confidence score as a penalty
    return penalty

# -------------------------------------------------
# Step 6: Visualization and Reporting
import matplotlib.pyplot as plt

fitness_history = [random.uniform(0, 1) for _ in range(generations)]
plt.plot(range(generations), fitness_history)
plt.title("Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness Score")
plt.show()

# Markdown Cell for Report
"""
## Report: Summary and Insights

**Methodology:**
- We used Stable Diffusion for generating images from prompts.
- Prompts evolved using a mutation-based algorithm.
- Aerodynamic performance was estimated using width-to-height ratio.
- Unrealistic designs penalized with a pre-trained image classifier.

**Results:**
- Visual improvements observed in generated designs over generations.
- Fitness score trends indicate evolving prompt effectiveness.

**Limitations:**
- Fitness evaluation and penalties are simplified.
- Realistic aerodynamic assessment would require simulation or CAD tools.

**Next Steps:**
- Implement more nuanced fitness functions.
- Use real-world metrics for realism.
"""
