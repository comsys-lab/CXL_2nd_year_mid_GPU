from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=str, required=False, help="Huggingface dataset directory (e.g., jenhsia/ragged)")
    parser.add_argument("--subset", type=str, default=None, help="Subset or config name (e.g., pubmed). Optional.")
    parser.add_argument("--llm", type=str, required=False, help="LLM to download (e.g., microsoft/DialoGPT-medium)")
    parser.add_argument("--embedding-model", type=str, required=False, help="Embedding model to download")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save the dataset or model")
    args = parser.parse_args()

    # download Datatset if hf_dir is provided
    if args.hf_dir:
        print(f"Downloading dataset from {args.hf_dir} to {args.save_dir}")
        # load dataset
        if args.subset:
            dataset = load_dataset(args.hf_dir, args.subset)
        else:
            dataset = load_dataset(args.hf_dir)

        # save dataset
        dataset.save_to_disk(args.save_dir)

    # Download LLM and tokenizer if llm is provided
    if args.llm:
        print(f"Downloading model {args.llm} to {args.save_dir}")
        try:
            # Download and save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.llm)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or "<pad>"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<pad>")
            tokenizer.save_pretrained(args.save_dir)
            
            # Download and save model
            model = AutoModelForCausalLM.from_pretrained(args.llm)
            model.save_pretrained(args.save_dir)
            
            print(f"Model and tokenizer saved to {args.save_dir}")
            
        except Exception as e:
            raise RuntimeError(f"Error downloading model or tokenizer: {e}")

    # Download embedding model if embedding-model is provided
    if args.embedding_model:
        print(f"Downloading embedding model {args.embedding_model} to {args.save_dir}")
        try:
            # Load and save the embedding model
            embedding_model = SentenceTransformer(args.embedding_model)
            embedding_model.save(args.save_dir)
            print(f"Embedding model saved to {args.save_dir}")
        except Exception as e:
            raise RuntimeError(f"Error downloading embedding model: {e}")

    # If no arguments are provided, print usage
    if not (args.hf_dir or args.llm or args.embedding_model):
        parser.print_help()
        sys.exit(1)
    
    print("Download complete.")