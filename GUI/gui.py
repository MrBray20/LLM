import dearpygui.dearpygui as dpg

models_available = [
    "Mistral-7B",
    "LLaMA 3 - 3B Instruct",
    "Phi-2",
    "GPT-J-6B",
    "DistilBERT",
    "TinyBERT"
]

def dummy_llm_output(model, task, text):
    if task == "Text Generation":
        return f"[{model}] Generated: {text}... (completed)"
    elif task == "Sentiment Analysis":
        sentiment = "Positive" if "good" in text.lower() else "Negative"
        return f"[{model}] Sentiment: {sentiment}"
    else:
        return f"[{model}] Unknown Task"

def run_comparison_callback():
    task = dpg.get_value("task_combo")
    input_text = dpg.get_value("input_text")
    selected_models = [model for model in models_available if dpg.get_value(f"checkbox_{model}")]

    if not input_text.strip():
        dpg.set_value("output_text", "Please enter some input text.")
        return
    if not selected_models:
        dpg.set_value("output_text", "Please select at least one model.")
        return

    results = [dummy_llm_output(model, task, input_text) for model in selected_models]
    dpg.set_value("output_text", "\n\n".join(results))

def select_all_callback(sender, app_data, user_data):
    is_checked = dpg.get_value(sender)
    for model in models_available:
        dpg.set_value(f"checkbox_{model}", is_checked)

def individual_model_callback(sender, app_data, user_data):
    # Jika ada model yang tidak dicentang, uncheck "Select All"
    for model in models_available:
        if not dpg.get_value(f"checkbox_{model}"):
            dpg.set_value("select_all", False)
            return
    # Jika semua dicentang, centang "Select All"
    dpg.set_value("select_all", True)

# GUI Setup
dpg.create_context()
dpg.create_viewport(title='LLM Comparison Tool', width=600, height=700)

with dpg.window(label="LLM Comparison Tool", width=580, height=680):
    dpg.add_text("Select Task:")
    dpg.add_combo(("Text Generation", "Sentiment Analysis"), default_value="Text Generation", tag="task_combo")

    dpg.add_spacer(height=10)
    dpg.add_text("Select Models to Compare:")

    dpg.add_checkbox(label="Select All Models", tag="select_all", callback=select_all_callback)

    for model in models_available:
        dpg.add_checkbox(label=model, tag=f"checkbox_{model}", callback=individual_model_callback)

    dpg.add_spacer(height=10)
    dpg.add_text("Enter Input Text:")
    dpg.add_input_text(tag="input_text", multiline=True, height=100, width=500, default_value="This product is really good!")

    dpg.add_spacer(height=10)
    dpg.add_button(label="Compare Models", callback=run_comparison_callback)

    dpg.add_spacer(height=10)
    dpg.add_text("Comparison Output:")
    dpg.add_input_text(tag="output_text", multiline=True, readonly=True, height=200, width=500)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
