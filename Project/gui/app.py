import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import pyperclip
import time
import re

class App():

    def __init__(self, runner):
        self.runner = runner
        dpg.create_context()
        self.models = ["Mistral", "LLaMA", "Gemma"]

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
        self.model_name_text_tags = {
            "mistral": "model_name_mistral",
            "llama": "model_name_llama",
            "gemma": "model_name_gemma"
        }
        self.copy_button_tags = {
            "mistral": "copy_button_Mistral",
            "llama": "copy_button_LLaMA",
            "gemma": "copy_button_Gemma"
        }
        self.loading_spinner_tags = {
            "mistral": "loading_Mistral",
            "llama": "loading_LLaMA",
            "gemma": "loading_Gemma"
        }
        self.status_text_tags = {
            "mistral": "status_text_Mistral",
            "llama": "status_text_LLaMA",
            "gemma": "status_text_Gemma"
        }

        # self.current_task = "Sentiment Analysis"
        self.current_task = "Text Generation"
        dpg.create_viewport(title="LLM Application", width=1350, height=800) # Increased height
        dpg.setup_dearpygui()
        self.add_font() # Load fonts before creating the UI
        self.run_app()
        dpg.show_viewport()
        dpg.set_viewport_resize_callback(self.update_layout)
        dpg.set_primary_window("main", True)
        self.app_theme()
        self.update_layout()
        
        dpg.start_dearpygui()
        dpg.destroy_context()

    def run_app(self):
        with dpg.window(tag="main"):
            # Header Section
            # Increased font size for the main title and slightly adjusted color
            dpg.add_text("LLM Comparison Tool", tag="main_title", color=[255, 220, 0, 255]) 
            dpg.add_separator()
            with dpg.group(horizontal=True, width=-1):
                dpg.add_spacer(width=dpg.get_viewport_width() * 0.5) # Spacer to push task combo to the right
                dpg.add_text("Select Task:")
                dpg.add_combo(("Text Generation", "Sentiment Analysis"), default_value=self.current_task, tag="task_combo", callback=self.task_combo_callback)
            
            dpg.add_separator()
            dpg.add_spacer(height=15)

            with dpg.child_window(label="model_selection_input_area", height=220, border=True):
                dpg.add_text("Configure Your Comparison:", color=[150, 200, 255, 255])
                dpg.add_separator()
                dpg.add_spacer(height=10)

                with dpg.group(horizontal=True):
                    dpg.add_text("Select Models:", color=[200, 200, 200, 255])
                    for model in self.models:
                        dpg.add_checkbox(label=model, tag=f"checkbox_{model}", callback=self.individual_model_callback)
                    dpg.add_checkbox(label="Select All Models", tag="select_all", callback=self.select_all_callback)

                dpg.add_spacer(height=15)
                dpg.add_text("Enter Input Text:", color=[200, 200, 200, 255])
                dpg.add_input_text(tag="input_text", multiline=True, width=-1, height=80, hint="Type your text here for analysis or generation...")

                dpg.add_spacer(height=10)
                
            dpg.add_button(label="Compare Models", callback=self.run_comparison_callback, width=-1, height=30)
            dpg.add_spacer(height=20)
            # Increased font size for the output comparison title and slightly adjusted color
            dpg.add_text("LLMs Output Comparison:", tag="output_title", color=[255, 220, 0, 255]) 
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Results Group (will be populated dynamically)
            with dpg.group(horizontal=True, tag="results_group"):
                self.update_output_layout_by_task()
            
            # Bind the larger font to the main titles
            dpg.bind_item_font("main_title", "large_font")
            dpg.bind_item_font("output_title", "large_font")

            
    def app_theme(self):
        with dpg.theme(tag="app_theme"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (20, 20, 20, 255)) # Darker background
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 40, 40, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 60, 60, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (90, 90, 90, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (120, 120, 120, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (30, 30, 30, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (80, 80, 80, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (0, 150, 200, 255)) # For selectable items, trees etc.
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (0, 180, 230, 255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (0, 100, 150, 255))

            with dpg.theme_component(item_type=dpg.mvInputText):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (50, 50, 50, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))

            with dpg.theme_component(item_type=dpg.mvButton):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5) # Rounded buttons
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 8)

        dpg.bind_theme("app_theme")

    def update_output_layout_by_task(self):
        # Clear existing items in results_group
        children = dpg.get_item_children("results_group", 1)
        if children:
            for child in children:
                dpg.delete_item(child)

        if self.current_task == "Sentiment Analysis":
            self.add_sentiment_output_layout()
        else: # Text Generation
            self.add_textgen_output_layout()

        
        time.sleep(0.01)
        self.update_layout()


    def add_textgen_output_layout(self):
        for model in self.models:
            tag_child_window = f"child_{model.lower()}"
            tag_output_text = self.model_tags_output[model.lower()]
            tag_model_name_text = self.model_name_text_tags[model.lower()]
            tag_copy_button = self.copy_button_tags[model.lower()]
            tag_loading_spinner = self.loading_spinner_tags[model.lower()]
            tag_status_text = self.status_text_tags[model.lower()]

            with dpg.child_window(tag=tag_child_window, parent="results_group", border=True):
                with dpg.group(horizontal=True, tag=f"header_group_{model.lower()}"):
                    dpg.add_text(model, tag=tag_model_name_text, color=[200, 200, 255, 255])
                    dpg.add_button(label="Copy", tag=tag_copy_button, callback=self.copy_to_clipboard_callback, user_data=tag_output_text, width=50, show=False)

                dpg.add_separator()
                dpg.add_spacer(height=5)
                dpg.add_loading_indicator(tag=tag_loading_spinner, circle_count=18, style=1, radius=8, speed=1.5, color=[0, 150, 200, 255], show=False)
                dpg.add_text("Ready", tag=tag_status_text, show=True, color=[100, 100, 100, 255]) 

                # Use input_text for scrollable output
                dpg.add_text("...",tag=tag_output_text, wrap=0, show=False)


    def add_sentiment_output_layout(self):
        for model in self.models:
            tag_child_window = f"child_{model.lower()}"
            sentiment_tag = self.sentiment_tags[model.lower()]
            explanation_tag = self.explanation_tags[model.lower()]
            tag_loading_spinner = self.loading_spinner_tags[model.lower()]
            tag_model_name_text = self.model_name_text_tags[model.lower()]
            tag_status_text = self.status_text_tags[model.lower()]


            with dpg.child_window(tag=tag_child_window, height=300, parent="results_group", border=True):
                with dpg.group(horizontal=True, tag=f"header_group_{model.lower()}"):
                    dpg.add_text(model, tag=tag_model_name_text, color=[200, 200, 255, 255])

                dpg.add_separator()
                dpg.add_spacer(height=5)
                dpg.add_loading_indicator(tag=tag_loading_spinner, circle_count=18, style=1, radius=8, speed=1.5, color=[0, 150, 200, 255], show=False)
                dpg.add_text("Ready", tag=tag_status_text, show=True, color=[100, 100, 100, 255])


                dpg.add_text("Sentiment:", tag=f"label_sentiment_{model.lower()}", show=False, color=[180, 180, 255, 255])
                dpg.add_text("...", tag=sentiment_tag, wrap=0, show=False, color=[255, 255, 0, 255])
                dpg.add_separator()
                dpg.add_spacer(height=5)
                dpg.add_text("Explanation:", tag=f"label_explanation_{model.lower()}", show=False, color=[180, 180, 255, 255])

                dpg.add_text("...",tag=explanation_tag, wrap=0, show=False)


    def task_combo_callback(self, sender, app_data, user_data):
        self.current_task = app_data
        self.update_output_layout_by_task()

    def update_layout(self, sender=None, app_data=None, user_data=None):
        total_width = dpg.get_viewport_client_width()
        column_width = (total_width // 3) - 15
        
        for model in self.models:
            model_name = model.lower()
            tag_child_window = f"child_{model_name}"
            tag_copy_button = self.copy_button_tags[model_name]
            tag_model_name_text = self.model_name_text_tags[model_name]
            tag_loading_spinner = self.loading_spinner_tags[model_name]
            
            if dpg.does_item_exist(tag_child_window):
                dpg.configure_item(tag_child_window, width=column_width, height=-1)

                child_pos_x, child_pos_y = dpg.get_item_pos(tag_child_window)
                child_width = dpg.get_item_width(tag_child_window)
                child_height = dpg.get_item_height(tag_child_window)

                # Position the loading spinner and status text
                spinner_diameter = 40
                if child_width > 0:
                    spinner_x = (child_width / 2) - spinner_diameter
                    # Position relative to the top of the child window, below the model name/separator
                    dpg.set_item_pos(tag_loading_spinner, [spinner_x, 34 + 50])
                    # Position status text below spinner
                    dpg.set_item_pos(self.status_text_tags[model_name], [spinner_x + (spinner_diameter / 4), dpg.get_item_pos(tag_loading_spinner)[1] + spinner_diameter + 3])


                if self.current_task == "Text Generation":
                    if dpg.does_item_exist(tag_copy_button) and dpg.does_item_exist(tag_model_name_text):
                        copy_button_width = dpg.get_item_width(tag_copy_button)
                        model_name_x, model_name_y = dpg.get_item_pos(tag_model_name_text)
                        # Position copy button to the right of the model name in the header group
                        dpg.set_item_pos(tag_copy_button, [child_width - copy_button_width - 15, model_name_y])
        
        if dpg.does_item_exist("warning_modal"):
            viewport_width = dpg.get_viewport_width()
            viewport_height = dpg.get_viewport_height()

            window_width = dpg.get_item_width("warning_modal")
            window_height = dpg.get_item_height("warning_modal")
            
            x_pos = (viewport_width / 2) - (window_width / 2)
            y_pos = (viewport_height / 2) - (window_height / 2)
            
            dpg.set_item_pos("warning_modal", [x_pos, y_pos])



    def add_font(self):
        with dpg.font_registry():
            # Default font size
            default_font = dpg.add_font("gui/NotoSans-Medium.ttf", 24, tag="default_font")
            dpg.bind_font(default_font)
            # Larger font for titles
            dpg.add_font("gui/NotoSans-Medium.ttf", 32, tag="large_font") # Increased font size for titles

    def individual_model_callback(self, sender, app_data, user_data):
        all_checked = True
        for model in self.models:
            if not dpg.get_value(f"checkbox_{model}"):
                all_checked = False
                break
        dpg.set_value("select_all", all_checked)

    def select_all_callback(self, sender, app_data, user_data):
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
            self.show_warning_modal("Selection Warning", "No models selected for comparison. Please choose at least one model.")
            return

        # Improved input validation
        cleaned_input = input_text.strip()
        if not cleaned_input:
            self.show_warning_modal("Input Warning", "Input text cannot be empty or just spaces.")
            return

        if not any(char.isalnum() for char in cleaned_input):
            self.show_warning_modal("Input Warning", "Input text cannot consist only of special characters.")
            return
        
        
        if self.is_single_word_surrounded_by_special_chars(input_text):
            self.show_warning_modal("Input Warning", "Input text is too short for meaningful generation. Please provide more context.")
            return

        
        for model in self.models:
            model_lower = model.lower()
            # Reset UI elements for selected models
            if model_lower in selected_models:
                dpg.show_item(self.loading_spinner_tags[model_lower])
                dpg.set_value(self.status_text_tags[model_lower], "Processing...")
                

                if self.current_task == "Sentiment Analysis":
                    dpg.set_value(self.sentiment_tags.get(model_lower), "")
                    dpg.set_value(self.explanation_tags.get(model_lower), "")
                    dpg.hide_item(f"label_sentiment_{model_lower}")
                    dpg.hide_item(self.sentiment_tags.get(model_lower))
                    dpg.hide_item(f"label_explanation_{model_lower}")
                    dpg.hide_item(self.explanation_tags.get(model_lower))
                else:
                    dpg.hide_item(self.copy_button_tags[model_lower])
                    dpg.set_value(self.model_tags_output.get(model_lower), "")
                    dpg.hide_item(self.model_tags_output.get(model_lower))
            else:
                dpg.hide_item(self.loading_spinner_tags[model_lower])
                dpg.set_value(self.status_text_tags[model_lower], "Ready")
                
                if self.current_task == "Sentiment Analysis":
                    dpg.set_value(self.sentiment_tags.get(model_lower), "")
                    dpg.set_value(self.explanation_tags.get(model_lower), "")
                    dpg.hide_item(f"label_sentiment_{model_lower}")
                    dpg.hide_item(self.sentiment_tags.get(model_lower))
                    dpg.hide_item(f"label_explanation_{model_lower}")
                    dpg.hide_item(self.explanation_tags.get(model_lower))
                else:
                    dpg.hide_item(self.copy_button_tags[model_lower])
                    dpg.set_value(self.model_tags_output.get(model_lower), "")
                    dpg.hide_item(self.model_tags_output.get(model_lower))

        self.runner.run_all(on_result=self.update_result_view, models=selected_models, task=selected_task, prompt=input_text)

    def update_result_view(self, model_name, result):
        # Ensure the model_name is lowercased for dictionary lookups
        model_name_lower = model_name.lower()

        if dpg.does_item_exist(self.loading_spinner_tags[model_name_lower]):
            dpg.hide_item(self.loading_spinner_tags[model_name_lower])
        if dpg.does_item_exist(self.status_text_tags[model_name_lower]):
            dpg.hide_item(self.status_text_tags[model_name_lower])

        if self.current_task == "Sentiment Analysis":
            sentiment_tag = self.sentiment_tags.get(model_name_lower)
            explanation_tag = self.explanation_tags.get(model_name_lower)

            if sentiment_tag and explanation_tag:
                sentiment_label = result.get("sentiment", "N/A")
                explanation_text = result.get("explanation", "No explanation provided.")

                dpg.set_value(sentiment_tag, sentiment_label)
                dpg.set_value(explanation_tag, explanation_text)

                dpg.show_item(f"label_sentiment_{model_name_lower}")
                dpg.show_item(sentiment_tag)
                dpg.show_item(f"label_explanation_{model_name_lower}")
                dpg.show_item(explanation_tag)
            else:
                print(f"Sentiment or explanation tags for model '{model_name_lower}' not found.")
        else:
            tag_output_text = self.model_tags_output.get(model_name_lower)
            if tag_output_text and dpg.does_alias_exist(tag_output_text):
                dpg.set_value(tag_output_text, result)
                dpg.show_item(tag_output_text)
                dpg.show_item(self.copy_button_tags[model_name_lower]) # Show copy button after results
            else:
                print(f"Output tag for model '{model_name_lower}' not found.")

    def show_warning_modal(self, title, message):
        
        if dpg.does_item_exist("warning_modal"):
            dpg.delete_item("warning_modal")

        with dpg.window(label=title, modal=True, show=True, tag="warning_modal", autosize=True, no_resize=True, no_move=True):
            dpg.add_text(message)
            dpg.add_spacer(height=10)
            dpg.add_button(label="OK", width=-1, callback=lambda: dpg.delete_item("warning_modal"))
        
        time.sleep(0.01)
        dpg.set_frame_callback(dpg.get_frame_count() + 1, self.update_layout)

    def copy_to_clipboard_callback(self, sender, app_data, user_data):
        item_tag_to_copy = user_data

        if dpg.does_item_exist(item_tag_to_copy):
            text_to_copy = dpg.get_value(item_tag_to_copy)
            try:
                pyperclip.copy(text_to_copy)
                print(f"Text from '{item_tag_to_copy}' successfully copied to clipboard")
                
            except pyperclip.PyperclipException as e:
                print(f"Failed to copy text to clipboard: {e}. Please ensure you have a clipboard backend installed.")
                print("For Linux, try installing xclip or xsel.")
                print("For Windows, ensure Python can access the clipboard.")
                self.show_warning_modal("Clipboard Error", "Failed to copy text. Please ensure a clipboard utility is installed (e.g., xclip/xsel on Linux).")


    def is_single_word_surrounded_by_special_chars(self, text):
    # Hitung jumlah kata (berisi huruf/angka minimal 1 karakter)
        words = re.findall(r'\b\w+\b', text)
        
        if len(words) < 5:
            return True
        
        return False
