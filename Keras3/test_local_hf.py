import argparse
import transformers

TEST_EXAMPLES = [
        #"Lizzy has to ship 540 pounds of fish that are packed into 30-pound crates. If the shipping cost of each crate is $1.5, how much will Lizzy pay for the shipment?",
        #"A school choir needs robes for each of its 30 singers. Currently, the school has only 12 robes so they decided to buy the rest. If each robe costs $2, how much will the school spend?",
        "Peter has 25 apples to sell. He sells the first 10 for $1 each, 10 more for $0.75 each and the last 5 for $0.50 each. How much money does he make?",
        "Bea has $40. She wants to rent a bike for $4/hour. How many hours can she ride the bike?",
        ]

# Prompt template for the training data and the finetuning tests
PROMPT_TEMPLATE = "user: {instruction}\nmodel: {response}\n"

#PROMPT_TEMPLATE = "prompts: {instruction}\nresponses: {response}\n"

TEST_PROMPTS = [
        PROMPT_TEMPLATE.format(instruction=example, response="")
        for example in TEST_EXAMPLES
        ]

def test_transformers_model(
        model: transformers.Gemma3ForCausalLM,
        tokenizer: transformers.GemmaTokenizer,
) -> None:
    for prompt in TEST_PROMPTS:
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=256, do_sample=True, temperature=1.0, top_k=10, top_p=0.95)
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{output}\n{'- '*40}")


def main():
    parser = argparse.ArgumentParser(
        description="Test a local Hugging Face Gemma model."
    )
    parser.add_argument(
        "--huggingface_model_dir",
        type=str,
        required=True,
        help="Path to the directory containing the local Hugging Face model and tokenizer.",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.huggingface_model_dir}")
    model = transformers.Gemma3ForCausalLM.from_pretrained(
        args.huggingface_model_dir,
        local_files_only=True,
        device_map="auto",  # Library "accelerate" to auto-select GPU
    )

    tokenizer = transformers.GemmaTokenizer.from_pretrained(
        args.huggingface_model_dir, local_files_only=True
    )

    print("\n--- Running inference tests ---")
    test_transformers_model(model, tokenizer)
    print("\n--- Tests complete ---")

if __name__ == "__main__":
    main()
