from transformers import pipeline

# Load pre-trained GPT-2 model
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# Get headline/topic
headline = input("Enter a news headline/topic: ")

# Prompt
prompt = f"""News Topic: {headline}

Key news points:
-"""

# Generate content
result = generator(
    prompt,
    max_new_tokens=120,
    num_return_sequences=1,
    temperature=0.8,
    do_sample=True,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)

# Get generated text
article = result[0]["generated_text"]

# Display
print("\n" + "=" * 60)
print("GENERATED NEWS")
print("=" * 60)
print(article)
print("=" * 60)


