import dearpygui.dearpygui as dpg
import dearpygui.demo  as demo
import pyperclip

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
        
        self.current_task="Text Generation"
        # dpg.show_documentation()
        # dpg.show_imgui_demo()
        self.run_app()
        # dpg.show_style_editor()
        # demo.show_demo()
        dpg.show_item_registry()
        dpg.create_viewport(title="LLM Aplication", width=1250, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.update_layout()
        dpg.set_primary_window("main",True)
        dpg.start_dearpygui()
        dpg.destroy_context()
        
    def run_app(self):
        with dpg.window(tag="main"):
            
            
            dpg.add_text("Select Task:")
            dpg.add_combo(("Text Generation", "Sentiment Analysis"), default_value="Text Generation", tag="task_combo", callback=self.task_combo_callback)


            dpg.add_spacer(height=40)
            with dpg.child_window(label="Option_select",height=150):
                dpg.add_text("Select Models to Compare:")
                with dpg.group(horizontal=True):
                    for model in self.models:
                        dpg.add_checkbox(label=model, tag=f"checkbox_{model}", callback=self.individual_model_callback)
                    dpg.add_checkbox(label="Select All Models", tag="select_all", callback=self.select_all_callback)
                    
                with dpg.group():
                    dpg.add_text("Enter Input Text:")
                    dpg.add_input_text(tag="input_text", multiline=True)
                    
            dpg.add_spacer(height=10)
            dpg.add_button(label="Compare Models", callback=self.run_comparison_callback)
            dpg.add_text("Hasil LLM")
            dpg.set_viewport_resize_callback(lambda s, a, u: self.update_layout())
            
            
            with dpg.group(horizontal=True,tag="results_group"):
                self.add_textgen_output_layout()
                
            # with dpg.group(horizontal=True,tag="results_group"):
            #     with dpg.child_window(tag="child_mistral", height=300):
            #         dpg.add_text("Mistral")
            #         dpg.add_input_text(tag="output_Mistral", multiline=True, readonly=True, width=-1, no_horizontal_scroll=True)

            #     with dpg.child_window(tag="child_llama", height=300):
            #         dpg.add_text("LLaMA")
            #         dpg.add_input_text(tag="output_LLaMA", multiline=True, readonly=True, width=-1, no_horizontal_scroll=True)

            #     with dpg.child_window(tag="child_gemma", height=300):
            #         dpg.add_text("Gemma")
            #         dpg.add_input_text(tag="output_Gemma", multiline=True, readonly=True, width=-1, no_horizontal_scroll=True)

    
        self.add_font()

    def update_output_layout_by_task(self):
        # Hapus dulu semua item di results_group
        children = dpg.get_item_children("results_group", 1)
        if children:
            for child in children:
                dpg.delete_item(child)

        # Tambahkan layout baru
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
            
            with dpg.child_window(tag=tag_child_window, parent="results_group", border=True):
                with dpg.group(horizontal=True, tag=f"header_group_{model.lower()}"):
                    dpg.add_text(model,tag=tag_model_name_text)
                    dpg.add_button(label="Copy", tag=tag_copy_button, callback=self.copy_to_clipboard_callback, user_data=tag_output_text, width=50, show=False)
                dpg.add_text("...",tag=tag_output_text, wrap=0, show=False)
                
    
    def add_sentiment_output_layout(self):
        for model in self.models:
            tag = f"child_{model.lower()}"
            sentiment_tag = self.sentiment_tags[model.lower()]
            explanation_tag = self.explanation_tags[model.lower()]

            with dpg.child_window(tag=tag, height=300, parent="results_group"):
                dpg.add_text(model)
                dpg.add_text("Sentiment")
                dpg.add_input_text(tag=sentiment_tag, readonly=True, width=-1)
                dpg.add_text("Explanation")
                dpg.add_input_text(tag=explanation_tag, readonly=True, multiline=True, width=-1, height=150)    

    def task_combo_callback(self, sender, app_data, user_data):
        self.current_task = app_data
        self.update_output_layout_by_task()
        self.update_layout() 
    
    def update_result(self, model_name, result):
        if self.current_task == "Sentiment Analysis":
            # Expecting result as dict with label and explanation
            sentiment_tag = self.sentiment_tags.get(model_name)
            explanation_tag = self.explanation_tags.get(model_name)
            if sentiment_tag and explanation_tag:
                dpg.set_value(sentiment_tag, result.get("label", "-"))
                dpg.set_value(explanation_tag, self.auto_wrap_text(result.get("explanation", "-")))
        else:
            # Text generation
            tag = self.model_tags_output.get(model_name)
            if tag:
                dpg.set_value(tag, self.auto_wrap_text(result))
            
    def update_layout(self, sender=None, app_data=None, user_data=None):
        total_width = dpg.get_viewport_client_width()
        column_width = total_width // 3 - 10  # sedikit padding antar kolom

        dpg.configure_item("child_mistral", width=column_width)
        dpg.configure_item("child_llama", width=column_width)
        dpg.configure_item("child_gemma", width=column_width)    


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

        if not input_text.strip():
            print("Input kosong, tidak menjalankan model.")
            return

        # if selected_task != "Text Generation":
        #     print(f"Task '{selected_task}' belum didukung.")
        #     return
        
        
        
        selected_models = [
            model.lower() for model in self.models
            if dpg.get_value(f"checkbox_{model}")
        ]
        
        if not selected_models:
            print("Tidak ada model yang dipilih.")
            return
        
        for model in selected_models:
            model_lower = model.lower()
            
            
            if self.current_task == "Sentiment Analysis":
                sentiment_tag = self.sentiment_tags.get(model_lower)
                explanation_tag = self.explanation_tags.get(model_lower)
                if sentiment_tag and explanation_tag:
                    dpg.set_value(sentiment_tag, "")
                    dpg.set_value(explanation_tag, "")
            else:
                output_tag = self.model_tags_output.get(model_lower)
                if output_tag:
                    dpg.set_value(output_tag, "")
        
        # def update_result(model_name, result):
        #     tag = self.model_tags_output.get(model_name)
        #     if tag and dpg.does_alias_exist(tag):
        #         dpg.set_value(tag,result)
        #     else:
        #         print(f"Tag untuk model '{model_name}' tidak ditemukan.")    
        self.runner.run_all(on_result=self.update_result_view, models=selected_models,task=selected_task, prompt=input_text)
    
    def update_result_view(self, model_name, result):
        
        if self.current_task == "Sentiment Analysis":
            sentimen_tag = self.sentiment_tags.get(model_name)
            explanation_tag = self.explanation_tags.get(model_name)
            
            if sentimen_tag and explanation_tag:
                sentimen_label = result.get("sentiment","-")
                explanation_text = result.get("explanation","-")

                dpg.set_value(sentimen_tag, sentimen_label)
                wrapped = auto_wrap_text(explanation_text)
                dpg.set_value(explanation_tag,wrapped)
            else:
                print(f"Tag sentiment atau explanation untuk model '{model_name}' tidak ditemukan.")
        else:
            tag_output_text = self.model_tags_output.get(model_name)
            if tag_output_text and dpg.does_alias_exist(tag_output_text):
                # wrapped = auto_wrap_text(result)
                dpg.set_value(tag_output_text, auto_wrap_text(result))
                dpg.show_item(tag_output_text)
                dpg.show_item(self.copy_button_tags[model_name])
            else:
                print(f"Tag untuk model '{model_name}' tidak ditemukan.")
        
    
    
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


