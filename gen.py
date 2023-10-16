import os
import uuid
import replicate
import numpy as np
import urllib.request
from PIL import Image
from io import BytesIO

def generate_images(prompts: list = [], 
                    height: int = 1024, 
                    width: int = 1024) -> list:
    outputs = []
    # Generate a unique id for this generation session
    session_id = uuid.uuid4()
    for i, prompt in enumerate(prompts):
        kwargs = {
            "model_version": "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
            "input": {
                "width": width,
                "height": height,
                "prompt": prompt,
                "seed": 42,
                "refine": "expert_ensemble_refiner",
                "scheduler": "KarrasDPM",
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "high_noise_frac": 0.8,
                "prompt_strength": 0.8,
                "num_inference_steps": 50
            }
        }
        output = replicate.run(**kwargs)
        # Download the image and convert it to a numpy array
        with urllib.request.urlopen(output[0]) as url:
            f = BytesIO(url.read())
        img = Image.open(f)

        # Save the image locally
        save_path = os.path.join('/home/oop/dev/data', f'{session_id}_{i+1}of{len(prompts)}.png')
        img.save(save_path)
        outputs.append((save_path, prompt))

    return outputs

if __name__ == "__main__":
    subject = "white bengal cat sitting down"  # You can replace this with any other subject
    # Test the function locally
    prompts = [
        # DALLE3 prompts
        # f"Dynamic 3/4 view of {subject} facing left with a clear emphasis on depth and perspective.",
        # f"Sharp 3/4 view of {subject} facing right, capturing the subtle contours and details.",
        # f"Direct frontal view of {subject}, offering a symmetrical and straightforward perspective.",
        # f"Top-down aerial view of {subject}, showcasing the subject from a bird's-eye perspective.",
        # f"Distinct profile view of {subject} facing left, highlighting the side silhouette.",
        # f"Isometric view of {subject}, creating a three-dimensional effect.",
        # f"Low-angle view of {subject}, looking up, emphasizing its dominance and stature.",
        # f"High-angle view of {subject}, looking down, making the subject appear smaller or more vulnerable.",
        # f"Rear view of {subject}, focusing on the back and hindquarters, capturing a unique perspective.",
        # f"Diagonal front-left view of {subject}, blending both frontal and profile features.",
        # Human prompts
        f"Dynamic 3/4 view of {subject}",
        f"Sharp 3/4 view of {subject}",
        f"Direct frontal view of {subject}",
        f"Top-down aerial view of {subject}",
        f"Distinct profile view of {subject}",
        f"Isometric view of {subject}",
        f"Low-angle view of {subject}",
        f"High-angle view of {subject}",
        f"Rear view of {subject}",
        f"Diagonal front-left view of {subject}",


    ]
    result = generate_images(prompts, height=512, width=512)
    print(result)