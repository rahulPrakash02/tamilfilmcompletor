import json

INPUT_FILE = "data/sample_tamil_movies.jsonl"
OUTPUT_FILE = "data/sft_train.jsonl"

SYSTEM = "You are a helpful Tamil film story writer. Write engaging, spoiler-free plots in 120–200 words."

PROMPT_TEMPLATE = (
    "<|system|>\n{system}\n<|user|>\n"
    "Actor: {actor}\nGenres: {genres}\n"
    "Task: Write a concise plot for a Tamil film starring the actor and fitting the genres.\n"
    "<|assistant|>\n"
)

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            row = json.loads(line)
            actor = row["actor"]
            genres = ", ".join(row["genres"])
            plot = row["plot"]

            prompt = PROMPT_TEMPLATE.format(system=SYSTEM, actor=actor, genres=genres)
            rec = {
                "prompt": prompt,
                "response": plot
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()