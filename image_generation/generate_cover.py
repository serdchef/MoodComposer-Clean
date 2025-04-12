from diffusers import StableDiffusionPipeline
import torch

def generate_cover_from_emotion(emotion):
    HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxx"  # Replace this with your real token

    prompt = f"A {emotion} classical music album cover in baroque style, oil painting, soft lighting"

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_auth_token=HF_TOKEN
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()

    image = pipe(prompt).images[0]
    image_path = f"{emotion}_cover.png"
    image.save(image_path)
    return image_path
