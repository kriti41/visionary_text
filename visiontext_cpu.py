#cpu friendly
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# CPU setup
device = "cpu"
model_id = "segmind/tiny-sd"

# Scheduler
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# Pipeline (no feature extractor needed)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float32,
    safety_checker=None
)
pipe = pipe.to(device)

# Test generation
prompt = "a cute cat, anime style"
image = pipe(prompt, height=128, width=128, num_inference_steps=10).images[0]

# Show image
image.show()
