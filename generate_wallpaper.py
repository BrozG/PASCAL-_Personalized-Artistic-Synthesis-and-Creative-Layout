import torch
from diffusers import DiffusionPipeline
import os
from datetime import datetime
import numpy as np

class WallpaperGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1-base", output_folder="output", low_vram_mode=False, hf_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Set CUDA memory configurations
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Load the diffusion pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            token=hf_token
        ).to(self.device)

        # Enable xFormers and attention slicing for better efficiency
        try:
            if self.device == "cuda":
                self.pipe.enable_xformers_memory_efficient_attention()
                print("Enabled xformers for better memory efficiency.")
        except Exception as e:
            print("xformers couldn't be enabled.", e)

        if low_vram_mode:
            print("Running in low VRAM mode...")
            self.pipe.enable_attention_slicing()

        os.makedirs(output_folder, exist_ok=True)
        self.output_folder = output_folder
    def generate_wallpaper(self, prompt: str) -> str:
        """Generates and saves a wallpaper based on the input prompt."""
        print(f"Generating image for prompt: '{prompt}'")

        try:
            # Generate the image
            with torch.cuda.amp.autocast() if self.device == "cuda" else torch.no_grad():
                output = self.pipe(prompt)
                image = output.images[0]

                # Normalize and clamp pixel values to avoid invalid casts
                image_array = np.array(image) / 255.0
                image_array = np.clip(image_array, 0.0, 1.0)
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

        # Save the image with a unique timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_folder, f"generated_wallpaper_{timestamp}.png")
        image.save(save_path)

        print(f"Wallpaper generated and saved at {save_path}!")
        return save_path


if __name__ == "__main__":
    low_vram = input("Run in low VRAM mode? (yes/no): ").strip().lower() == "yes"
    hf_token = ["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"]

    generator = WallpaperGenerator(
        output_folder=r"C:\\broz_files\\project\\PASCAL _Personalized Artistic Synthesis and Creative Layout\\output",
        low_vram_mode=low_vram,
        hf_token=hf_token
    )

    prompt = input("Enter the wallpaper prompt: ")
    generator.generate_wallpaper(prompt)
