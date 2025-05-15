# from lm_eval import evaluator

# evaluator.simple_evaluate(
#     model="hf",
#     model_args="pretrained=D:\huggingface_cache\hub\models--unsloth--mistral-7b-instruct-v0.3-bnb-4bit",
#     tasks=["mathqa"],  # cukup ganti di sini
#     batch_size=1,
#     device="cuda"
# )

# from huggingface_hub import snapshot_download

# path = snapshot_download("unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
# print(path)

from lm_eval import evaluator
import json
import numpy as np
model_path = (
    r"D:\huggingface_cache\hub\models--unsloth--mistral-7b-instruct-v0.3-bnb-4bit"
    r"\snapshots\d5f623888f1415cf89b5c208d09cb620694618ee"
)

# results = evaluator.simple_evaluate(
#     model="hf",
#     model_args=(
#         f"pretrained={model_path},"
#         "trust_remote_code=True"
#     ),
#     tasks=["lambada"],  # Ganti/expand daftar task jika ingin
#     batch_size=1,
#     device="cuda",  # Atau "cpu" jika tidak ada GPU
#     limit=1
# )


results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_path}",
    tasks=["lambada_openai"],
    batch_size=1,
    max_batch_size=1,
    limit=2,  # batasi agar tidak OOM di GPU kecil
    device="cuda"
)

# Fungsi untuk mengonversi objek yang tidak bisa diserialisasi ke string
def convert_obj_to_serializable(obj):
    if isinstance(obj, (np.generic, np.ndarray)):  # cek tipe data numpy
        return obj.item()  # mengubah menjadi tipe dasar seperti int atau float
    return str(obj)  # mengubah objek lain ke string

# Menggunakan json.dumps dengan custom default
print(json.dumps(results, default=convert_obj_to_serializable, indent=2))

