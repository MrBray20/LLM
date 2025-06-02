import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import time
import threading
import queue
import pyperclip
import re

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
                        result = {"label": "Neutral", "explanation": "The sentiment in this text is largely neutral, presenting facts without strong emotional language. There. are no clear indicators of positive or negative feelings. This text is generally balanced and objective, providing information without overt emotional bias. It sticks to factual reporting and avoids factual reporting and avoids strong opinions, ensuring a calm and measured tone."}
                    elif model == "gemma":
                        result = {"label": "Negative", "explanation": "A clearly negative sentiment is conveyed here, with words expressing dissatisfaction and criticism. The tone suggests displeasure or disappointment. This text highlights various shortcomings and issues, emphasizing drawbacks rather than benefits. The choice of vocabulary strongly indicates disapproval and a critical stance."}
                    else:
                        result = {"label": "Unknown", "explanation": "No sentiment detected for this model."}
                else: # Text Generation
                    result = f"Output from {model.capitalize()} for prompt: '{prompt}'. This is a simulated long text to demonstrate wrapping and loading indicators in DearPyGui. It showcases how different LLMs might respond to a given input, providing unique perspectives or generated content that can be easily compared within the application interface."
                
                gui_queue.put((on_result, (model, result))) 
            
            gui_queue.put((dpg.set_value, ("input_text", ""))) 

        # Start the simulation in a new thread
        threading.Thread(target=_simulate_run).start()

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
        # dpg.show_item_registry() # Opsional: uncomment untuk melihat registry item
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        dpg.set_viewport_resize_callback(self.update_layout) 
        dpg.set_start_callback(self.update_layout) 
        
        dpg.set_frame_callback(dpg.get_frame_count() + 1, self.process_gui_queue_loop) 

        dpg.set_primary_window("main",True)
        dpg.start_dearpygui()
        dpg.destroy_context()

    def process_gui_queue_loop(self): 
        """Memproses antrean GUI untuk pembaruan UI dari thread lain."""
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
        """Membangun tata letak utama aplikasi."""
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
            # Tombol "Compare Models" yang akan memicu perbandingan
            dpg.add_button(label="Compare Models", callback=self.run_comparison_callback, width=-1, tag="compare_button") 
            dpg.add_text("Hasil LLM")
            
            # Grup untuk menampung hasil perbandingan model
            with dpg.group(horizontal=True, tag="results_group"):
                self.add_textgen_output_layout() 
            
            # self.add_font() # Uncomment jika Anda memiliki file font

    def update_output_layout_by_task(self):
        """Memperbarui tata letak output berdasarkan tugas yang dipilih."""
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
        """Menambahkan tata letak untuk output generasi teks."""
        for model in self.models:
            tag_child_window = f"child_{model.lower()}"
            tag_output_text = self.model_tags_output[model.lower()] 
            tag_loading_spinner = self.loading_spinner_tags[model.lower()]
            tag_copy_button = self.copy_button_tags[model.lower()] 
            tag_model_name_text = self.model_name_text_tags[model.lower()] 

            with dpg.child_window(tag=tag_child_window, parent="results_group", border=True):
                with dpg.group(horizontal=True, tag=f"header_group_{model.lower()}"): 
                    dpg.add_text(model, tag=tag_model_name_text) 
                    dpg.add_button(
                        label="Copy", 
                        tag=tag_copy_button, 
                        callback=self.copy_to_clipboard_callback, 
                        user_data=tag_output_text, 
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
        """Menambahkan tata letak untuk output analisis sentimen."""
        for model in self.models:
            tag_child_window = f"child_{model.lower()}"
            sentiment_tag = self.sentiment_tags[model.lower()]
            explanation_tag = self.explanation_tags[model.lower()]
            tag_loading_spinner = self.loading_spinner_tags[model.lower()]
            tag_copy_button = self.copy_button_tags[model.lower()] 
            tag_model_name_text = self.model_name_text_tags[model.lower()] 

            with dpg.child_window(tag=tag_child_window, parent="results_group", border=True):
                with dpg.group(horizontal=True, tag=f"header_group_{model.lower()}"):
                    dpg.add_text(model, tag=tag_model_name_text) 
                    dpg.add_button(
                        label="Copy", 
                        tag=tag_copy_button, 
                        callback=self.copy_to_clipboard_callback, 
                        user_data=explanation_tag, 
                        show=False, # Sembunyikan awalnya untuk sentimen
                        width=60 
                    )
                
                dpg.add_loading_indicator(tag=tag_loading_spinner, circle_count=12, radius=20, speed=1.5, color=[0, 150, 200, 255], show=False)
                
                dpg.add_text("Sentiment:", tag=f"label_sentiment_{model.lower()}", show=False)
                dpg.add_text("...", tag=sentiment_tag, wrap=0, show=False) 
                dpg.add_text("Explanation:", tag=f"label_explanation_{model.lower()}", show=False)
                dpg.add_text("...", tag=explanation_tag, wrap=0, show=False) 
                
                dpg.hide_item(f"label_sentiment_{model.lower()}")
                dpg.hide_item(sentiment_tag)
                dpg.hide_item(f"label_explanation_{model.lower()}")
                dpg.hide_item(explanation_tag)

    def task_combo_callback(self, sender, app_data, user_data):
        """Callback saat pilihan tugas di combo box berubah."""
        self.current_task = app_data
        self.update_output_layout_by_task()

    def update_result(self, model_name, result):
        """Memperbarui UI dengan hasil dari model."""
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
        else: # Text Generation
            tag_output_text = self.model_tags_output.get(model_lower)
            if tag_output_text and dpg.does_item_exist(tag_output_text):
                dpg.set_value(tag_output_text, auto_wrap_text(result))
                dpg.show_item(tag_output_text) 
                dpg.show_item(self.copy_button_tags[model_lower])
            else:
                print(f"Tag untuk model '{model_name}' tidak ditemukan (update_result).")
            
    def update_layout(self, sender=None, app_data=None, user_data=None):
        """Memperbarui tata letak saat jendela diubah ukurannya atau pada startup."""
        total_width = dpg.get_viewport_client_width()
        column_width = (total_width // 3) - 15 

        spinner_diameter = 40 

        # Mendapatkan posisi y dari elemen penting untuk menghitung tinggi yang tersedia
        # Perlu dipastikan item "main" dan "results_group" ada
        main_window_pos_y = dpg.get_item_pos("main")[1] if dpg.does_item_exist("main") else 0
        results_group_pos_y = dpg.get_item_pos("results_group")[1] if dpg.does_item_exist("results_group") else 0

        viewport_height = dpg.get_viewport_client_height()
        bottom_margin = 20 

        available_height_for_results = viewport_height - results_group_pos_y - bottom_margin
        
        if dpg.does_item_exist("results_group"):
            dpg.configure_item("results_group", height=available_height_for_results)
        
        for model in self.models:
            model_lower = model.lower()
            tag_child_window = f"child_{model_lower}"
            tag_loading_spinner = self.loading_spinner_tags[model_lower]
            tag_copy_button = self.copy_button_tags[model_lower]
            tag_model_name_text = self.model_name_text_tags[model_lower]
            
            if dpg.does_item_exist(tag_child_window):
                dpg.configure_item(tag_child_window, width=column_width, height=-1) 

                child_pos_x, child_pos_y = dpg.get_item_pos(tag_child_window)
                child_width = dpg.get_item_width(tag_child_window)
                child_height = dpg.get_item_height(tag_child_window) 
                
                # Posisi spinner di tengah child window
                if child_width > 0 and child_height > 0: 
                    spinner_x = child_pos_x + (child_width / 2) - (spinner_diameter / 2)
                    spinner_y = child_pos_y + (child_height / 2) - (spinner_diameter / 2)
                    
                    if dpg.does_item_exist(tag_loading_spinner):
                        dpg.set_item_pos(tag_loading_spinner, [spinner_x, spinner_y])

                # Posisi tombol "Copy" agar rata kanan di dalam child window
                if dpg.does_item_exist(tag_copy_button) and dpg.does_item_exist(tag_model_name_text):
                    model_name_x, model_name_y = dpg.get_item_pos(tag_model_name_text)
                    copy_button_width = dpg.get_item_width(tag_copy_button) 
                    
                    copy_button_x = child_pos_x + child_width - copy_button_width - 10 
                    copy_button_y = model_name_y 

                    dpg.set_item_pos(tag_copy_button, [copy_button_x, copy_button_y])
                                
    def add_font(self):
        """Mencoba memuat font kustom."""
        try:
            with dpg.font_registry():
                # Sesuaikan path font jika diperlukan
                a = dpg.add_font("gui/NotoSans-Medium.ttf", 16) 
            dpg.bind_font(a)
        except Exception as e:
            print(f"Failed to load font: {e}. Skipping font loading.")

    def individual_model_callback(self,sender, app_data, user_data):
        """Callback untuk checkbox model individual, mengatur checkbox 'Select All'."""
        for model in self.models:
            if not dpg.get_value(f"checkbox_{model}"):
                dpg.set_value("select_all", False)
                return
        dpg.set_value("select_all", True)
        
    def select_all_callback(self,sender, app_data, user_data):
        """Callback untuk checkbox 'Select All Models'."""
        is_checked = dpg.get_value(sender)
        for model in self.models:
            dpg.set_value(f"checkbox_{model}", is_checked)
            
    def show_warning_modal(self, title, message):
        """Menampilkan jendela modal peringatan."""
        # Pastikan jendela sebelumnya (jika ada) sudah dihapus
        if dpg.does_item_exist("warning_modal"):
            dpg.delete_item("warning_modal")

        # Buat jendela modal baru
        with dpg.window(label=title, modal=True, show=True, tag="warning_modal", 
                        autosize=True, no_resize=True, no_move=True):
            dpg.add_text(message)
            dpg.add_spacer(height=10)
            dpg.add_button(label="OK", width=-1, callback=lambda: dpg.delete_item("warning_modal"))
            
            # Posisikan jendela di tengah layar setelah dibuat dan ukurannya ditentukan
            # Ini mungkin memerlukan satu frame untuk mendapatkan ukuran akurat
            # Oleh karena itu, penempatan di callback ini adalah perkiraan terbaik
            # atau bisa dipindahkan ke proses_gui_queue_loop dengan event khusus
            viewport_width = dpg.get_viewport_width()
            viewport_height = dpg.get_viewport_height()
            
            # Mendapatkan ukuran jendela modal (mungkin belum final di frame yang sama)
            window_width = dpg.get_item_width("warning_modal")
            window_height = dpg.get_item_height("warning_modal")
            
            x_pos = (viewport_width / 2) - (window_width / 2)
            y_pos = (viewport_height / 2) - (window_height / 2)
            dpg.set_item_pos("warning_modal", [x_pos, y_pos])


    def run_comparison_callback(self, sender, app_data, user_data):
        """Callback utama untuk menjalankan perbandingan model."""
        selected_task = dpg.get_value("task_combo")
        input_text = dpg.get_value("input_text")

        # --- VALIDASI INPUT DIMULAI DI SINI ---
        # 1. Validasi untuk input kosong atau hanya spasi
        if not input_text or not input_text.strip():
            self.show_warning_modal("Peringatan Input", "Input teks tidak boleh kosong atau hanya berisi spasi.")
            return 
        
        # 2. Validasi untuk input hanya karakter khusus (tanpa huruf/angka sama sekali)
        if not any(char.isalnum() for char in input_text):
            self.show_warning_modal("Peringatan Input", "Input teks tidak boleh hanya terdiri dari karakter khusus.")
            return

        # 3. Validasi untuk input berupa satu kata dikelilingi/diikuti karakter khusus
        if is_single_word_surrounded_by_special_chars(input_text):
            self.show_warning_modal("Peringatan Input", "Input teks terlalu singkat atau hanya terdiri dari satu kata yang dikelilingi karakter khusus.")
            return
        
        # 4. Validasi untuk jumlah kata minimal
        if count_words(input_text) <= 5:
            self.show_warning_modal("Peringatan Input", "Input teks harus lebih dari 5 kata.")
            return
        # --- VALIDASI INPUT SELESAI DI SINI ---
            
        selected_models = [
            model.lower() for model in self.models
            if dpg.get_value(f"checkbox_{model}")
        ]
        
        # --- PENAMBAHAN VALIDASI MODEL KOSONG ---
        if not selected_models:
            self.show_warning_modal("Peringatan Pilihan Model", "Tidak ada model yang dipilih untuk dibandingkan.")
            return
        # --- AKHIR PENAMBAHAN VALIDASI MODEL KOSONG ---

        # Sembunyikan output dan tampilkan spinner untuk model yang akan dijalankan
        for model in self.models:
            model_lower = model.lower()
            tag_child_window = f"child_{model_lower}"
            
            if model_lower in selected_models: 
                if dpg.does_item_exist(tag_child_window):
                    dpg.show_item(tag_child_window)

                # Sembunyikan tombol copy dan output/sentiment/explanation di awal setiap eksekusi
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
                # Sembunyikan child window untuk model yang tidak dipilih
                if dpg.does_item_exist(tag_child_window):
                    dpg.hide_item(tag_child_window)

        # Jalankan runner model di thread terpisah
        self.runner.run_all(on_result=self.update_result, models=selected_models, task=selected_task, prompt=input_text)
        
    def copy_to_clipboard_callback(self, sender, app_data, user_data):
        """Menyalin teks dari item yang ditentukan ke clipboard."""
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
    """Fungsi pembantu untuk memecah teks panjang menjadi beberapa baris."""
    text = text.replace('\n', ' ')
    wrapped = ''
    while len(text) > max_length:
        wrap_pos = text.rfind(' ', 0, max_length)
        if wrap_pos == -1: # Tidak ada spasi dalam max_length, paksa potong
            wrap_pos = max_length
        wrapped += text[:wrap_pos] + '\n'
        text = text[wrap_pos:].lstrip()
    wrapped += text
    return wrapped

def is_single_word_surrounded_by_special_chars(text):
    """
    Memeriksa apakah string hanya mengandung satu kata alfanumerik
    yang dikelilingi atau diikuti oleh karakter non-alfanumerik.
    Contoh: "halo!!!", "test@@@", "!@#kata", "123++"
    """
    # Mencari pola:
    # ^\W* : Dimulai dengan nol atau lebih karakter non-kata (\W)
    # (\w+) : Diikuti oleh SATU atau lebih karakter kata (\w), ini adalah "kata" kita
    # \W*$ : Diikuti oleh nol atau lebih karakter non-kata (\W) sampai akhir string
    # re.IGNORECASE: Mengabaikan perbedaan huruf besar/kecil
    match = re.fullmatch(r"^\W*(\w+)\W*$", text, re.IGNORECASE)
    
    # Jika ada kecocokan dan "kata" yang ditemukan tidak kosong
    if match:
        word = match.group(1) # Ambil bagian kata yang cocok
        # Pastikan kata itu sendiri mengandung setidaknya satu karakter alfanumerik
        return any(char.isalnum() for char in word)
    return False

def count_words(text):
    """
    Menghitung jumlah kata dalam sebuah string.
    Kata didefinisikan sebagai urutan karakter alfanumerik.
    """
    words = re.findall(r'\b\w+\b', text)
    return len(words)

if __name__ == "__main__":
    app_runner = MockRunner() 
    app = App(app_runner)
