import os
import uuid
import replicate
import numpy as np
import urllib.request
from PIL import Image
from io import BytesIO

def generate_images(prompts: list = [], height: int = 1024, width: int = 1024, save_images: bool = False) -> np.ndarray:
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
        img_array = np.array(img)
        outputs.append(img_array)

        # Save the image locally if save_images is True
        if save_images:
            save_path = os.path.join(os.getenv('DATA_DIR', '.'), f'{session_id}_{i+1}of{len(prompts)}.png')
            img.save(save_path)

    # Stack all images into a single numpy tensor
    return np.stack(outputs, axis=0)

if __name__ == "__main__":
    # Test the function locally
    prompts = [
        "3/4 portrait of cat facing left",
        "3/4 portrait of cat facing right"
    ]
    result = generate_images(prompts, height=512, width=512, save_images=True)
    print(result)