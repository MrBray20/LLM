import dearpygui.dearpygui as dpg
import dearpygui.demo  as demo
import pyperclip
import time
import re
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
        self.model_name_text_tags={
            "mistral": "model_name_mistral",
            "llama":"model_name_llama",
            "gemma":"model_name_gemma"
        }
        self.copy_button_tags={
            "mistral": "copy_button_Mistral",
            "llama": "copy_button_LLaMA",
            "gemma": "copy_button_Gemma"
        }
        self.loading_spinner_tags = {
            "mistral": "loading_Mistral",
            "llama": "loading_LLaMA",
            "gemma": "loading_Gemma"
        }
        
        # self.current_task="Text Generation"
        self.current_task="Sentiment Analysis"
        # dpg.show_documentation()
        # dpg.show_imgui_demo()
        self.run_app()
        # dpg.show_style_editor()
        # demo.show_demo()
        dpg.show_item_registry()
        dpg.set_viewport_resize_callback(self.update_layout)
        dpg.create_viewport(title="LLM Aplication", width=1250, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.update_layout()
        dpg.set_primary_window("main",True)
        dpg.start_dearpygui()
        dpg.destroy_context()
        
    def run_app(self):
        with dpg.window(tag="main"):
            with dpg.group(horizontal=True):
                dpg.add_text("Select Task:")
                dpg.add_combo(
                    ("Text Generation", "Sentiment Analysis"), 
                    default_value=self.current_task, 
                    tag="task_combo", 
                    callback=self.task_combo_callback,
                    width=250 
                )
            dpg.add_spacer(height=15)
            with dpg.child_window(label="model_selection_input_area",height=200):
                dpg.add_text("Select Models to Compare:")
                with dpg.group(horizontal=True):
                    for model in self.models:
                        dpg.add_checkbox(label=model, tag=f"checkbox_{model}", callback=self.individual_model_callback)
                    dpg.add_checkbox(label="Select All Models", tag="select_all", callback=self.select_all_callback)
                    
                with dpg.group():
                    dpg.add_text("Enter Input Text:")
                    dpg.add_input_text(tag="input_text", multiline=True, width=-1, height=100)
                    
            dpg.add_spacer(height=10)
            dpg.add_button(label="Compare Models", callback=self.run_comparison_callback,width=-1)
            dpg.add_text("LLMs Output")
            
            
            with dpg.group(horizontal=True,tag="results_group"):
                self.update_output_layout_by_task()
    
        self.add_font()

    def update_output_layout_by_task(self):
        # Hapus dulu semua item di results_group
        children = dpg.get_item_children("results_group", 1)
        if children:
            for child in children:
                dpg.delete_item(child)

        if self.current_task == "Sentiment Analysis":
            self.add_sentiment_output_layout()
        else:
            self.add_textgen_output_layout()
        
        
    def add_textgen_output_layout(self):
        for model in self.models:
            tag_child_window = f"child_{model.lower()}"
            tag_output_text = self.model_tags_output[model.lower()]
            tag_model_name_text = self.model_name_text_tags[model.lower()]
            tag_copy_button= self.copy_button_tags[model.lower()]
            tag_loading_spinner = self.loading_spinner_tags[model.lower()]
            
            with dpg.child_window(tag=tag_child_window, parent="results_group", border=True):
                with dpg.group(horizontal=True, tag=f"header_group_{model.lower()}"):
                    dpg.add_text(model,tag=tag_model_name_text)
                    dpg.add_button(label="Copy", tag=tag_copy_button, callback=self.copy_to_clipboard_callback, user_data=tag_output_text, width=50, show=False)
                
                dpg.add_loading_indicator(tag=tag_loading_spinner, circle_count=18, style=1, radius=8, speed=1.5, color=[0, 150, 200, 255], show=False)
                
                dpg.add_text("...",tag=tag_output_text, wrap=0, show=False)
                
    
    def add_sentiment_output_layout(self):
        for model in self.models:
            tag_child_window = f"child_{model.lower()}"
            sentiment_tag = self.sentiment_tags[model.lower()]
            explanation_tag = self.explanation_tags[model.lower()]
            tag_loading_spinner = self.loading_spinner_tags[model.lower()]
            tag_model_name_text = self.model_name_text_tags[model.lower()]
            
            with dpg.child_window(tag=tag_child_window, height=300, parent="results_group"):
                with dpg.group(horizontal=True, tag=f"header_group_{model.lower()}"):
                    dpg.add_text(model, tag=tag_model_name_text)  

                dpg.add_loading_indicator(tag=tag_loading_spinner, circle_count=18, style=1, radius=8, speed=1.5, color=[0, 150, 200, 255], show=False)
            
                dpg.add_text("Sentiment:", tag=f"label_sentiment_{model.lower()}", show=False)
                dpg.add_text("...", tag=sentiment_tag, wrap=0, show=False) # Wrap akan diatur di update_result
                dpg.add_text("Explanation:", tag=f"label_explanation_{model.lower()}", show=False)
                dpg.add_text("...", tag=explanation_tag, wrap=0, show=False) # Wrap akan diatur di update_result
                

    def task_combo_callback(self, sender, app_data, user_data):
        self.current_task = app_data
        self.update_output_layout_by_task()
        time.sleep(0.001)
        self.update_layout()
        
            
    def update_layout(self, sender=None, app_data=None, user_data=None):
        total_width = dpg.get_viewport_client_width()
        column_width = total_width // 3 - 10  
        
        for model in self.models:
            model_name = model.lower()
            tag_child_window = f"child_{model_name}"
            tag_copy_button = self.copy_button_tags[model_name]
            tag_model_name_text = self.model_name_text_tags[model_name]
            tag_loading_spinner = self.loading_spinner_tags[model_name]
            # print(dpg.get_item_pos(tag_copy_button))
            dpg.configure_item(tag_child_window, width=column_width, height =-1)
            # print(model_name)
            
            child_pos_x, child_pos_y = dpg.get_item_pos(tag_child_window)
            child_width = dpg.get_item_width(tag_child_window)
            child_height = dpg.get_item_height(tag_child_window) 
            
            if self.current_task == "Text Generation":
                copy_button_width = dpg.get_item_width(tag_copy_button)
                model_name_x, model_name_y = dpg.get_item_pos(tag_model_name_text)

                spinner_diameter = 40
                
                if child_width > 0: 
                    spinner_x = (child_width / 2) - spinner_diameter
                    
                    if dpg.does_item_exist(tag_loading_spinner):
                        dpg.set_item_pos(tag_loading_spinner, [spinner_x, 34 + 50])                
                
                if dpg.does_item_exist(tag_model_name_text):
                    child_width = dpg.get_item_width(tag_child_window)
                    dpg.set_item_pos(tag_copy_button,[child_width - copy_button_width - 15,model_name_y])
                    
            elif self.current_task == "Sentiment Analysis":
                spinner_diameter = 40
                
                if child_width > 0: 
                    spinner_x = (child_width / 2) - spinner_diameter
                    
                    if dpg.does_item_exist(tag_loading_spinner):
                        dpg.set_item_pos(tag_loading_spinner, [spinner_x, 34 + 50])      

            if dpg.does_item_exist("warning_modal"):
                
                viewport_width = dpg.get_viewport_width()
                viewport_height = dpg.get_viewport_height()
        
                window_width = dpg.get_item_width("warning_modal")
                window_height = dpg.get_item_height("warning_modal")
            
                x_pos = (viewport_width / 2) - (window_width /2)
                y_pos = (viewport_height / 2) - (window_height /2)
                dpg.set_item_pos("warning_modal", [x_pos, y_pos])
                
    def add_font(self):
        with dpg.font_registry():
            a= dpg.add_font("gui/NotoSans-Medium.ttf",16)
        dpg.bind_font(a)
        
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

        selected_models = [
            model.lower() for model in self.models
            if dpg.get_value(f"checkbox_{model}")
        ]
        
        if not selected_models:
            self.show_warning_modal("Peringatan Pilihan Model", "Tidak ada model yang dipilih untuk dibandingkan.")
            return
        
        if not input_text or not input_text.strip():
            self.show_warning_modal("Peringatan Input", "Input teks tidak boleh kosong atau hanya berisi spasi.")
            return 
        
        if not any(char.isalnum() for char in input_text):
            self.show_warning_modal("Peringatan Input", "Input teks tidak boleh hanya terdiri dari karakter khusus.")
            return
        
        if self.is_single_word_surrounded_by_special_chars(input_text):
            self.show_warning_modal("Peringatan Input", "Input teks terlalu singkat atau hanya terdiri dari satu kata yang dikelilingi karakter khusus.")
            return
        
        if not selected_models:
            print("Tidak ada model yang dipilih.")
            return
        
        for model in selected_models:
            model_lower = model.lower()
            
            if dpg.does_item_exist(self.copy_button_tags[model_lower]):
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
            else:
                output_tag = self.model_tags_output.get(model_lower)
                if output_tag:
                    dpg.set_value(output_tag, "")
                    dpg.hide_item(output_tag)
            dpg.show_item(self.loading_spinner_tags[model_lower])
            
            
        self.runner.run_all(on_result=self.update_result_view, models=selected_models,task=selected_task, prompt=input_text)
    
    def update_result_view(self, model_name, result):
        
        if dpg.does_item_exist(self.loading_spinner_tags[model_name]):
            dpg.hide_item(self.loading_spinner_tags[model_name])
        
        if self.current_task == "Sentiment Analysis":
            sentiment_tag = self.sentiment_tags.get(model_name)
            explanation_tag = self.explanation_tags.get(model_name)
            
            if sentiment_tag and explanation_tag:
                sentimen_label = result.get("sentiment","-")
                explanation_text = result.get("explanation","-")

                dpg.set_value(sentiment_tag, sentimen_label)
                dpg.set_value(explanation_tag,explanation_text)
                
                dpg.show_item(f"label_sentiment_{model_name}")
                dpg.show_item(sentiment_tag)
                dpg.show_item(f"label_explanation_{model_name}")
                dpg.show_item(explanation_tag)
            else:
                print(f"Tag sentiment atau explanation untuk model '{model_name}' tidak ditemukan.")
            
        else:
            tag_output_text = self.model_tags_output.get(model_name)
            if tag_output_text and dpg.does_alias_exist(tag_output_text):
                dpg.set_value(tag_output_text, result)
                dpg.show_item(tag_output_text)
                dpg.show_item(self.copy_button_tags[model_name])
            else:
                print(f"Tag untuk model '{model_name}' tidak ditemukan.")
                
    def show_warning_modal(self, title, message):
        
        if dpg.does_item_exist("warning_modal"):
            dpg.delete_item("warning_modal")

        
        with dpg.window(label=title, modal=True, show=True, tag="warning_modal", autosize=True, no_resize=True, no_move=True):
            dpg.add_text(message)
            dpg.add_spacer(height=10)
            dpg.add_button(label="OK", width=-1, callback=lambda: dpg.delete_item("warning_modal"))
        
        time.sleep(0.01)
        
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        
        window_width = dpg.get_item_width("warning_modal")
        window_height = dpg.get_item_height("warning_modal")
    
        x_pos = (viewport_width / 2) - (window_width /2)
        y_pos = (viewport_height / 2) - (window_height /2)
        dpg.set_item_pos("warning_modal", [x_pos, y_pos])
    
    def copy_to_clipboard_callback(self, sender, app_data, user_data):
        item_tag_to_copy = user_data
        
        if dpg.does_item_exist(item_tag_to_copy):
            text_to_copy = dpg.get_value(item_tag_to_copy)
            try:
                pyperclip.copy(text_to_copy)
                print(f"Teks dari '{item_tag_to_copy}' berhasil disalin ke clipboard")
            except pyperclip.PyperclipException as e:
                print(f"Gagal menyalin teks ke clipboard: {e}. Pastikan Anda memiliki backend clipboard yang terinstal.")
                print("Untuk Linux, coba install xclip atau xsel.")
                print("Untuk Windows, pastikan Python dapat mengakses clipboard.")
                
    def is_single_word_surrounded_by_special_chars(self,text):
        match = re.fullmatch(r"^\W*(\w+)\W*$", text, re.IGNORECASE)
        
        # Jika ada kecocokan dan "kata" yang ditemukan tidak kosong
        if match:
            word = match.group(1) # Ambil bagian kata yang cocok
            # Pastikan kata itu sendiri mengandung setidaknya satu karakter alfanumerik
            return any(char.isalnum() for char in word)
        return False

    def count_words(text):
        words = re.findall(r'\b\w+\b', text)
        return len(words)




