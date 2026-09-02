import openai
import numpy as np
import os

# Configuración de la clave API de OpenAI
api_key = "sk-"
openai.api_key = api_key

# Rutas de los archivos de entrada y salida
data_path = r"\\150.244.56.196\Compartir\NLP\NLPEGNA\code\data"
database_file = "FairCVdb.npy"
save_path = r"\\150.244.56.196\Compartir\NLP\NLPCV\Anonimization\ChatGPT\ChatGPT4\Test"
output_file = "anonymized_fairCVCHATGPTPER4.npy"
output_path = os.path.join(save_path, output_file)

# Cargar el archivo de la base de datos de biografías
fairCV = np.load(os.path.join(data_path, database_file), allow_pickle=True).item()
bios_train = fairCV['Bios Test']

# Cargar el archivo existente de resultados (si existe), o crear una lista vacía
if os.path.exists(output_path):
    anonimizado_bios_train = list(np.load(output_path, allow_pickle=True))
    start_index = len(anonimizado_bios_train)
    print(f"Archivo cargado con {start_index} biografías procesadas.")
else:
    anonimizado_bios_train = []
    start_index = 0
    print("No se encontró el archivo existente. Comenzando desde cero.")

# Procesar cada biografía desde el índice guardado
for i in range(start_index, len(bios_train)):
    bio = bios_train[i]
    original_text = bio[0]  # Texto original de la biografía
    input_text = f"""Act as a name anonymizer. Replace proper names of people in the sentences with [MASK], without modifying titles, pronouns, or other parts of the sentence. Do not alter city names, articles, or prepositions, only personal names. Provide only the modified sentence without explanations or quotation marks.

    Examples:
    - "John met Sarah at the coffee shop" -> "[MASK] met [MASK] at the coffee shop"
    - "Emma and Robert went on vacation" -> "[MASK] and [MASK] went on vacation"

    Your turn: {original_text}
    """

    try:
        # Llamada a la API de OpenAI para anonimizar el texto
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": input_text}
            ],
            max_tokens=300
        )

        # Obtener la respuesta generada (texto anonimizado)
        anonimizado_text = response.choices[0].message['content'].strip()

    except Exception as e:
        print(f"Error al procesar la biografía {i}: {e}")
        anonimizado_text = original_text  # Fallback en caso de error

    # Guardar la biografía original y la anonimizada en la lista
    anonimizado_bio = [original_text, anonimizado_text]
    anonimizado_bios_train.append(anonimizado_bio)

    # Guardar el archivo actualizado en cada iteración
    np.save(output_path, anonimizado_bios_train)
    print(f"Biografía {i + 1} procesada y guardada.")

print("Procesamiento completado.")
