from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen-7b-8k-inst", trust_remote_code = True)

model = AutoModelForCausalLM.from_pretrained("Salesforce/xgen-7b-8k-inst", torch_dtype = torch.bfloat16, from_tf=True)

def summarize(text):
    header = ("A chat between a curious human and an artificial intelligence assistant. Tpip install tiktokenhe assistant gives helpful, detailed, and polite answers to the human's questions.\n\n")
    text = header + "### Human: Summarize the following article. \n\n" + text + "\n###"
    inputs = tokenizer(text, return_tensors = "pt")
    generated_ids = model.generare(**inputs, max_length = 1024, do_sample = True, top_k = 50, top_p = 0.95, temperature = 0.7)
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens = True).lstrip()
    summary = summary.split("### Assistant:")[1]
    summary = summary.split("<|endoftext|>")[0]
    return gr.Textbox(value = summary)


with gr.Blocks() as demo:
    with gr.Row():
        textbox = gr.Textbox(lines = 20, label="Text")
        summary = gr.Textbox(label = "Summary", lines = 20)
    submit = gr.Button(text = "Summarize")
    submit.click(summarize, inputs = textbox, outputs = summary)

demo.launch()