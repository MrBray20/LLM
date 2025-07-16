import dearpygui.dearpygui as dpg

class App():
    
    def __init__(self):
        dpg.create_context()
        self.models=["Mistral","LLaMA","Gemma"]
        self.model_tags_output = {
            "mistral": "output_Mistral",
            "llama": "output_LLaMA",
            "gemma": "output_Gemma"
        }
        self.model_name_text_tags = {
            "mistral": "model_name_Mistral",
            "llama": "model_name_LLaMA",
            "gemma": "model_name_Gemma"
        }

        self.current_task="Text Generation"
        
        dpg.create_viewport(title="LLM Application", width=1250, height=600)
        self.run_app() 
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.show_font_manager()
        dpg.set_viewport_resize_callback(self.update_layout) 
        dpg.set_start_callback(self.update_layout) 

        dpg.set_primary_window("main",True)
        dpg.start_dearpygui()
        dpg.destroy_context()

        
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
            dpg.add_spacer(height=50) 
            with dpg.child_window(tag="model_selection_input_area", height=150, autosize_x=True, border=True): 
                dpg.add_text("Select Models to Compare:")
                with dpg.group(horizontal=True):
                    for model in self.models:
                        dpg.add_checkbox(label=model, tag=f"checkbox_{model}", default_value=True) 
                    dpg.add_checkbox(label="Select All Models", tag="select_all", default_value=True)
                    
                dpg.add_spacer(height=10)
                dpg.add_text("Enter Input Text:")
                dpg.add_input_text(tag="input_text", multiline=True, width=-1, height=60)
                    
            dpg.add_spacer(height=10)
            dpg.add_button(label="Compare Models", width=-1, tag="compare_button") 
            dpg.add_text("Hasil LLM")
            
            with dpg.group(horizontal=True, tag="results_group"):
                self.add_textgen_output_layout() 
                
    def add_textgen_output_layout(self):
        """Menambahkan tata letak untuk output generasi teks."""
        for model in self.models:
            tag_child_window = f"child_{model.lower()}"
            tag_output_text = self.model_tags_output[model.lower()] 
            tag_model_name_text = self.model_name_text_tags[model.lower()] 

            with dpg.child_window(tag=tag_child_window, parent="results_group", border=True):
                with dpg.group(horizontal=True, tag=f"header_group_{model.lower()}"): 
                    dpg.add_text(model, tag=tag_model_name_text) 
                
                dpg.add_text("...", tag=tag_output_text, wrap=0, show=False )

    def task_combo_callback(self, sender, app_data, user_data):
        """Callback saat pilihan tugas di combo box berubah."""
        self.current_task = app_data
        self.update_output_layout_by_task()

            
    def update_layout(self, sender=None, app_data=None, user_data=None):
        """Memperbarui tata letak saat jendela diubah ukurannya atau pada startup."""
        total_width = dpg.get_viewport_client_width()
        column_width = (total_width // 3) - 15 

        spinner_diameter = 40 
        results_group_pos_y = dpg.get_item_pos("results_group")[1] if dpg.does_item_exist("results_group") else 0

        viewport_height = dpg.get_viewport_client_height()
        bottom_margin = 20 

        available_height_for_results = viewport_height - results_group_pos_y - bottom_margin
        
        if dpg.does_item_exist("results_group"):
            dpg.configure_item("results_group", height=available_height_for_results)
        
        for model in self.models:
            model_lower = model.lower()
            tag_child_window = f"child_{model_lower}"
            
            if dpg.does_item_exist(tag_child_window):
                dpg.configure_item(tag_child_window, width=column_width, height=-1) 

                child_pos_x, child_pos_y = dpg.get_item_pos(tag_child_window)
                child_width = dpg.get_item_width(tag_child_window)
                child_height = dpg.get_item_height(tag_child_window) 
                
                if child_width > 0 and child_height > 0: 
                    spinner_x = child_pos_x + (child_width / 2) - (spinner_diameter / 2)
                    spinner_y = child_pos_y + (child_height / 2) - (spinner_diameter / 2)
                                

if __name__ == "__main__":
    app = App()
