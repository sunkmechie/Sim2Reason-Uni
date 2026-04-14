import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def convert_to_hf(checkpoint_path, output_dir, push_to_hub=False, repo_id=None):
    # Load adapter config (contains base model name)
    print("🔍 Loading adapter config...")
    peft_config = PeftConfig.from_pretrained(checkpoint_path)

    print(f"📦 Loading base model: {peft_config.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype="auto",
        device_map="auto"
    )

    # Load LoRA adapter
    print(f"🧩 Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    # Merge LoRA into base weights
    print("🔗 Merging adapter into base model...")
    model = model.merge_and_unload()

    # Load tokenizer
    print("📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Save merged model
    print(f"💾 Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Optional: upload to Hugging Face Hub
    if push_to_hub:
        if repo_id is None:
            raise ValueError("❌ --repo_id must be provided when --push_to_hub is set.")
        from huggingface_hub import HfApi
        HfApi().create_repo(repo_id, private=False, exist_ok=True)
        print(f"☁️ Uploading to Hugging Face Hub: {repo_id}")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print("✅ Upload complete!")

def main():
    parser = argparse.ArgumentParser(description="Convert LoRA SFT model to Hugging Face format.")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to LoRA SFT checkpoint directory.")
    parser.add_argument("--output_dir", required=True, help="Path to save the merged Hugging Face model.")
    parser.add_argument("--push_to_hub", action="store_true", help="Flag to upload model to Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, help="Repo ID on Hugging Face Hub (e.g., username/model-name)")

    args = parser.parse_args()

    convert_to_hf(
        checkpoint_path=args.checkpoint_dir,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id
    )

if __name__ == "__main__":
    main()