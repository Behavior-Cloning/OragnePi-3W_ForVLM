from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from PIL import Image
import requests


def main():
    load_path = "."

    # load the processor
    print("Loading processor")
    processor = AutoProcessor.from_pretrained(
        load_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # load the model
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # process the image and text
    print("Processing...")
    inputs = processor.process(
        images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
        text="Describe this image."
    )

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    print("Generating....")
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # print the generated text
    print(generated_text)



if __name__ == '__main__':
    main()