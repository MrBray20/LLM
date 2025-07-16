import dearpygui.dearpygui as dpg
dpg.create_context()
with dpg.window(tag="main"):
    pass
dpg.create_viewport(title="LLM Application", width=1250, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main",True)
dpg.start_dearpygui()
dpg.destroy_context()
