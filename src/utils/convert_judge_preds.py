import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Process model completions and export cleaned generations.")
    parser.add_argument("--subset", type=str, default="medmcqa", choices=["medqa", "medmcqa", "mmlu"], help="Dataset subset name (e.g., medmcqa)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name used for saving output paths")
    parser.add_argument("--input_path", type=str, required=True, help="Path to judges JSONL file")
    parser.add_argument("--modes", nargs="+", default=["open"], help="Modes to extract (default: open)")

    args = parser.parse_args()
    
    # Load input
    with open(args.input_path, "r") as f:
        data = [json.loads(line) for line in f]

    # Process each mode
    for mode in args.modes:
        out_dir = f"out/completions/{args.model_name}/{args.subset}"
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"generations_{mode}.jsonl")
        backup_path = os.path.join(out_dir, f"generations_{mode}_old.jsonl")

        # If output exists, rename it
        if os.path.exists(out_path):
            # Remove previous backup if exists to avoid accumulation
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(out_path, backup_path)
            print(f"[i] Existing file detected. Renamed to {backup_path}")

        # Write fresh output file
        with open(out_path, "w") as f_out:
            for item in data:
                out_dict = {
                    "id_question": item["question_id"],
                    "mode": mode,
                    "gold_answer": item["gold_answer"],
                    "final_answer": item.get(f"{mode}_response"),
                    "correct": item["gold_answer"] == item.get(f"{mode}_response")
                }

                if mode == "open" and "valid_alternative" in item:
                    out_dict["valid_alternative"] = item["valid_alternative"]

                json.dump(out_dict, f_out)
                f_out.write("\n")

        print(f"[✓] Saved new {mode} results to {out_path}")


if __name__ == "__main__":
    main()
