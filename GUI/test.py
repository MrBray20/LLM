import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import time
import threading
import queue
import pyperclip # Import the pyperclip library

# Pastikan Anda sudah menginstal pyperclip: pip install pyperclip

# Create a thread-safe queue for GUI updates
gui_queue = queue.Queue()

# Placeholder untuk runner Anda, sesuaikan dengan implementasi sebenarnya
class MockRunner:
    def run_all(self, on_result, models, task, prompt):
        def _simulate_run():
            for model in models:
                time.sleep(1.5) # Simulate processing time
                if task == "Sentiment Analysis":
                    if model == "mistral":
                        result = {"label": "Positive", "explanation": "This text expresses a very positive sentiment towards the subject matter. The language used is uplifting and enthusiastic, suggesting strong approval and satisfaction."}
                    elif model == "llama":
                        result = {"label": "Neutral", "explanation": "The sentiment in this text is largely neutral, presenting facts without strong emotional language. There are no clear indicators of positive or negative feelings."}
                    elif model == "gemma":
                        result = {"label": "Negative", "explanation": "A clearly negative sentiment is conveyed here, with words expressing dissatisfaction and criticism. The tone suggests displeasure or disappointment."}
                    else:
                        result = {"label": "Unknown", "explanation": "No sentiment detected for this model."}
                else: # Text Generation
                    result = f"Output from {model.capitalize()} for prompt: '{prompt}'. This is a simulated long text to demonstrate wrapping and loading indicators in DearPyGui. It showcases how different LLMs might respond to a given input, providing unique perspectives or generated content that can be easily compared within the application interface."
                
                gui_queue.put((on_result, (model, result))) 
            
            gui_queue.put((dpg.set_value, ("input_text", ""))) 

        thread = threading.Thread(target=_simulate_run)
        thread.start()

class App():
    
    def __init__(self,runner):
        self.runner= runner
        dpg.create_context()
        self.models=["Mistral","LLaMA","Gemma"]
        self.model_tags_output = {
            "mistral": "output_Mistral",
            "llama": "output_LLaMA",
            "gemma": "output_Gemma"
        }
        self.sentiment_tags = {
            "mistral": "sentiment_Mistral",
            "llama": "sentiment_LLaMA",
            "gemma": "sentiment_Gemma"
        }
        self.explanation_tags = {
            "mistral": "explanation_Mistral",
            "llama": "explanation_LLaMA",
            "gemma": "explanation_Gemma"
        }
        self.loading_spinner_tags = {
            "mistral": "loading_Mistral",
            "llama": "loading_LLaMA",
            "gemma": "loading_Gemma"
        }
        self.copy_button_tags = {
            "mistral": "copy_button_Mistral",
            "llama": "copy_button_LLaMA",
            "gemma": "copy_button_Gemma"
        }
        # Tambahkan tag untuk nama model di tata letak agar bisa mendapatkan posisinya
        self.model_name_text_tags = {
            "mistral": "model_name_Mistral",
            "llama": "model_name_LLaMA",
            "gemma": "model_name_Gemma"
        }

        self.current_task="Text Generation"
        
        dpg.create_viewport(title="LLM Application", width=1250, height=600)
        self.run_app() 
        dpg.show_item_registry()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        dpg.set_viewport_resize_callback(self.update_layout) 
        dpg.set_start_callback(self.update_layout) 
        
        dpg.set_frame_callback(dpg.get_frame_count() + 1, self.process_gui_queue_loop) 

        dpg.set_primary_window("main",True)
        dpg.start_dearpygui()
        dpg.destroy_context()

    def process_gui_queue_loop(self): 
        while not gui_queue.empty():
            try:
                callback, args = gui_queue.get_nowait() 
                callback(*args)
            except queue.Empty:
                pass 
            except Exception as e:
                print(f"Error processing GUI queue task: {e}")
        
        if dpg.is_dearpygui_running(): 
            dpg.set_frame_callback(dpg.get_frame_count() + 1, self.process_gui_queue_loop)

        
    def run_app(self):
        with dpg.window(tag="main"):
            with dpg.group(horizontal=True):
                dpg.add_text("Select Task:")
                dpg.add_combo(
                    ("Text Generation", "Sentiment Analysis"), 
                    default_value="Text Generation", 
                    tag="task_combo", 
                    callback=self.task_combo_callback,
                    width=250 
                )
            dpg.add_spacer(height=20) 
            with dpg.child_window(tag="model_selection_input_area", height=150, autosize_x=True, border=True): 
                dpg.add_text("Select Models to Compare:")
                with dpg.group(horizontal=True):
                    for model in self.models:
                        dpg.add_checkbox(label=model, tag=f"checkbox_{model}", callback=self.individual_model_callback, default_value=True) 
                    dpg.add_checkbox(label="Select All Models", tag="select_all", callback=self.select_all_callback, default_value=True)
                    
                dpg.add_spacer(height=10)
                dpg.add_text("Enter Input Text:")
                dpg.add_input_text(tag="input_text", multiline=True, width=-1, height=60)
                    
            dpg.add_spacer(height=10)
            dpg.add_button(label="Compare Models", callback=self.run_comparison_callback, width=-1)
            dpg.add_text("Hasil LLM")
            
            with dpg.group(horizontal=True, tag="results_group"):
                self.add_textgen_output_layout() 
            
        # self.add_font() 

    def update_output_layout_by_task(self):
        children = dpg.get_item_children("results_group", 1) 
        if children:
            for child in children:
                dpg.delete_item(child)

        if self.current_task == "Sentiment Analysis":
            self.add_sentiment_output_layout()
        else:
            self.add_textgen_output_layout()
        
        self.update_layout() 

    def add_textgen_output_layout(self):
        for model in self.models:
            tag_child_window = f"child_{model.lower()}"
            tag_output_text = self.model_tags_output[model.lower()] 
            tag_loading_spinner = self.loading_spinner_tags[model.lower()]
            tag_copy_button = self.copy_button_tags[model.lower()] 
            tag_model_name_text = self.model_name_text_tags[model.lower()] # Tag untuk nama model

            with dpg.child_window(tag=tag_child_window, parent="results_group", border=True):
                # Buat grup horizontal untuk nama model dan tombol copy
                with dpg.group(horizontal=True, tag=f"header_group_{model.lower()}"): 
                    dpg.add_text(model, tag=tag_model_name_text) # Beri tag pada teks nama model
                    dpg.add_button(
                        label="Copy", 
                        tag=tag_copy_button, 
                        callback=self.copy_to_clipboard_callback, 
                        user_data=tag_output_text, 
                        # show=False,
                        # Atur lebar tombol agar konsisten, posisinya akan diatur manual
                        width=60 
                    )
                
                dpg.add_loading_indicator(tag=tag_loading_spinner, circle_count=12, radius=20, speed=1.5, color=[0, 150, 200, 255], show=False)
                
                dpg.add_text(
                    "...", 
                    tag=tag_output_text, 
                    wrap=0, 
                    show=False 
                )

    def add_sentiment_output_layout(self):
        for model in self.models:
            tag_child_window = f"child_{model.lower()}"
            sentiment_tag = self.sentiment_tags[model.lower()]
            explanation_tag = self.explanation_tags[model.lower()]
            tag_loading_spinner = self.loading_spinner_tags[model.lower()]
            tag_copy_button = self.copy_button_tags[model.lower()] 
            tag_model_name_text = self.model_name_text_tags[model.lower()] # Tag untuk nama model

            with dpg.child_window(tag=tag_child_window, parent="results_group", border=True):
                # Buat grup horizontal untuk nama model dan tombol copy
                with dpg.group(horizontal=True, tag=f"header_group_{model.lower()}"):
                    dpg.add_text(model, tag=tag_model_name_text) # Beri tag pada teks nama model
                    dpg.add_button(
                        label="Copy", 
                        tag=tag_copy_button, 
                        callback=self.copy_to_clipboard_callback, 
                        user_data=explanation_tag, 
                        # show=False,
                        width=60 
                    )
                
                dpg.add_loading_indicator(tag=tag_loading_spinner, circle_count=12, radius=20, speed=1.5, color=[0, 150, 200, 255], show=False)
                
                dpg.add_text("Sentiment", tag=f"label_sentiment_{model.lower()}", show=False)
                dpg.add_input_text(tag=sentiment_tag, readonly=True, width=-1, show=False)
                dpg.add_text("Explanation", tag=f"label_explanation_{model.lower()}", show=False)
                dpg.add_input_text(tag=explanation_tag, readonly=True, multiline=True, width=-1, height=100, show=False)
                
                dpg.hide_item(f"label_sentiment_{model.lower()}")
                dpg.hide_item(sentiment_tag)
                dpg.hide_item(f"label_explanation_{model.lower()}")
                dpg.hide_item(explanation_tag)

    def task_combo_callback(self, sender, app_data, user_data):
        self.current_task = app_data
        self.update_output_layout_by_task()

    def update_result(self, model_name, result):
        model_lower = model_name.lower()
        
        if dpg.does_item_exist(self.loading_spinner_tags[model_lower]):
            dpg.hide_item(self.loading_spinner_tags[model_lower])

        if self.current_task == "Sentiment Analysis":
            sentiment_tag = self.sentiment_tags.get(model_lower)
            explanation_tag = self.explanation_tags.get(model_lower)
            
            if sentiment_tag and explanation_tag and dpg.does_item_exist(sentiment_tag):
                sentiment_label = result.get("label", "-") 
                explanation_text = result.get("explanation", "-")

                dpg.set_value(sentiment_tag, sentiment_label)
                dpg.set_value(explanation_tag, auto_wrap_text(explanation_text))
                
                dpg.show_item(f"label_sentiment_{model_lower}")
                dpg.show_item(sentiment_tag)
                dpg.show_item(f"label_explanation_{model_lower}")
                dpg.show_item(explanation_tag)
                dpg.show_item(self.copy_button_tags[model_lower])
                # self.update_layout() # <--- ADDED THIS LINE
        else: # Text Generation
            tag_output_text = self.model_tags_output.get(model_lower)
            if tag_output_text and dpg.does_item_exist(tag_output_text):
                dpg.set_value(tag_output_text, auto_wrap_text(result))
                dpg.show_item(tag_output_text) 
                print(self.copy_button_tags[model_lower])
                dpg.show_item(self.copy_button_tags[model_lower])
                # self.update_layout() # <--- ADDED THIS LINE
            else:
                print(f"Tag untuk model '{model_name}' tidak ditemukan (update_result).")
            
    def update_layout(self, sender=None, app_data=None, user_data=None):
        total_width = dpg.get_viewport_client_width()
        column_width = (total_width // 3) - 15 

        spinner_diameter = 40 

        main_window_pos_y = dpg.get_item_pos("main")[1] 
        results_group_pos_y = dpg.get_item_pos("results_group")[1] 

        viewport_height = dpg.get_viewport_client_height()

        bottom_margin = 20 

        available_height_for_results = viewport_height - results_group_pos_y - bottom_margin
        
        if dpg.does_item_exist("results_group"):
            dpg.configure_item("results_group", height=available_height_for_results)
        print(self.models)
        for model in self.models:
            model_lower = model.lower()
            tag_child_window = f"child_{model_lower}"
            tag_loading_spinner = self.loading_spinner_tags[model_lower]
            tag_copy_button = self.copy_button_tags[model_lower]
            tag_model_name_text = self.model_name_text_tags[model_lower]
            print(tag_copy_button)
            if dpg.does_item_exist(tag_child_window):
                dpg.configure_item(tag_child_window, width=column_width, height=-1) 

                child_pos_x, child_pos_y = dpg.get_item_pos(tag_child_window)
                child_width = dpg.get_item_width(tag_child_window)
                child_height = dpg.get_item_height(tag_child_window) 
                
                # Posisi spinner
                if child_width > 0 and child_height > 0: 
                    spinner_x = child_pos_x + (child_width / 2) - (spinner_diameter / 2)
                    spinner_y = child_pos_y + (child_height / 2) - (spinner_diameter / 2)
                    
                    if dpg.does_item_exist(tag_loading_spinner):
                        dpg.set_item_pos(tag_loading_spinner, [spinner_x, spinner_y])

                # Posisi tombol "Copy"
                if dpg.does_item_exist(tag_copy_button) and dpg.does_item_exist(tag_model_name_text):
                    # Dapatkan posisi dan lebar item nama model
                    model_name_x, model_name_y = dpg.get_item_pos(tag_model_name_text)
                    model_name_width = dpg.get_item_width(tag_model_name_text)
                    
                    # Dapatkan lebar tombol copy
                    copy_button_width = dpg.get_item_width(tag_copy_button) # Ini akan menjadi 60 (dari add_button)
                    
                    # Hitung posisi X untuk tombol copy agar rata kanan dalam child_window
                    # Posisi X child_window + lebar child_window - lebar tombol - margin
                    copy_button_x = child_pos_x + child_width - copy_button_width - 10 # margin 10px dari kanan
                    
                    # Posisi Y tombol copy sama dengan posisi Y nama model
                    copy_button_y = model_name_y 

                    dpg.set_item_pos(tag_copy_button, [copy_button_x, copy_button_y])
                            
    def add_font(self):
        try:
            with dpg.font_registry():
                a = dpg.add_font("gui/NotoSans-Medium.ttf", 16) 
            dpg.bind_font(a)
        except Exception as e:
            print(f"Failed to load font: {e}. Skipping font loading.")

    def individual_model_callback(self,sender, app_data, user_data):
        for model in self.models:
            if not dpg.get_value(f"checkbox_{model}"):
                dpg.set_value("select_all", False)
                return
        dpg.set_value("select_all", True)
        
    def select_all_callback(self,sender, app_data, user_data):
        is_checked = dpg.get_value(sender)
        for model in self.models:
            dpg.set_value(f"checkbox_{model}", is_checked)
            
    def run_comparison_callback(self, sender, app_data, user_data):
        selected_task = dpg.get_value("task_combo")
        input_text = dpg.get_value("input_text")

        if not input_text.strip():
            print("Input kosong, tidak menjalankan model.")
            return
            
        selected_models = [
            model.lower() for model in self.models
            if dpg.get_value(f"checkbox_{model}")
        ]
        
        if not selected_models:
            print("Tidak ada model yang dipilih.")
            return

        for model in self.models:
            model_lower = model.lower()
            tag_child_window = f"child_{model_lower}"
            
            if model_lower in selected_models: 
                if dpg.does_item_exist(tag_child_window):
                    dpg.show_item(tag_child_window)

                # Always hide copy button and output/sentiment/explanation at the start of a new run
                # This ensures a clean slate and that they are only shown once results arrive
                dpg.hide_item(self.copy_button_tags[model_lower])

                if self.current_task == "Sentiment Analysis":
                    sentiment_tag = self.sentiment_tags.get(model_lower)
                    explanation_tag = self.explanation_tags.get(model_lower)
                    if sentiment_tag and explanation_tag:
                        dpg.set_value(sentiment_tag, "")
                        dpg.set_value(explanation_tag, "")
                        dpg.hide_item(f"label_sentiment_{model_lower}")
                        dpg.hide_item(sentiment_tag)
                        dpg.hide_item(f"label_explanation_{model_lower}")
                        dpg.hide_item(explanation_tag)
                else:  # Text Generation
                    output_tag = self.model_tags_output.get(model_lower)
                    if output_tag:
                        dpg.set_value(output_tag, "")
                        dpg.hide_item(output_tag) 
                
                dpg.show_item(self.loading_spinner_tags[model_lower])
            else:
                if dpg.does_item_exist(tag_child_window):
                    dpg.hide_item(tag_child_window)


        self.runner.run_all(on_result=self.update_result, models=selected_models, task=selected_task, prompt=input_text)
        
    def copy_to_clipboard_callback(self, sender, app_data, user_data):
        item_tag_to_copy = user_data 
        
        if dpg.does_item_exist(item_tag_to_copy):
            text_to_copy = dpg.get_value(item_tag_to_copy)
            try:
                pyperclip.copy(text_to_copy)
                print(f"Teks dari '{item_tag_to_copy}' berhasil disalin ke clipboard.")
            except pyperclip.PyperclipException as e:
                print(f"Gagal menyalin teks ke clipboard: {e}. Pastikan Anda memiliki backend clipboard yang terinstal.")
                print("Untuk Linux, coba install xclip atau xsel.")
                print("Untuk Windows, pastikan Python dapat mengakses clipboard.")

def auto_wrap_text(text, max_length=65):
    text = text.replace('\n', ' ')
    wrapped = ''
    while len(text) > max_length:
        wrap_pos = text.rfind(' ', 0, max_length)
        if wrap_pos == -1:
            wrap_pos = max_length
        wrapped += text[:wrap_pos] + '\n'
        text = text[wrap_pos:].lstrip()
    wrapped += text
    return wrapped

if __name__ == "__main__":
    app_runner = MockRunner() 
    app = App(app_runner)