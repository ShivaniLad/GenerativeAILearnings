from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('./models')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
tokenizer.pad_token = tokenizer.eos_token

prompt = ''


def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=30, pad_token_id=tokenizer.eos_token_id)
    generate_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generate_output


print(generate_text("Should I invest in stocks?"))
print(generate_text("Where can i go for this summer vacation? can you please suggest me a location?"))
print(generate_text("How can i save the environment from pollution?"))
