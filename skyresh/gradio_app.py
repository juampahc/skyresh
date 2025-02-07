import gradio as gr
import requests
from configuration import get_settings
from utils import render_html

# URL where FastAPI is running
BASE_SETTINGS = get_settings()
API_URL = "http://127.0.0.1:8000"

# Function for fetching the config of inference server
def fetch_config():
    """
    Fetch the current configuration from the FastAPI app.
    """
    global BASE_SETTINGS
    try:
        headers = {
        "Content-Type":"application/json",
        "access_token": BASE_SETTINGS.API_KEY
        }
        response = requests.get(f"{BASE_SETTINGS.API_INTERNAL_URL}/config", headers=headers)
        response.raise_for_status()
        config = response.json()
        # Map the config to the Gradio input order:
        # MODEL_ID, LOAD_VINO, LOAD_TOKENIZER, VINO_FILE.
        return config["MODEL_ID"], config["LOAD_VINO"], config["LOAD_TOKENIZER"], config["VINO_FILE"]
    except Exception as _:
        return "Error"

def update_config_gradio(model_id, load_vino, load_tokenizer, vino_file):
    """
    Send a POST request to update the configuration.
    """
    global BASE_SETTINGS
    headers = {
        "Content-Type":"application/json",
        "access_token": BASE_SETTINGS.API_KEY
        }
    payload = {
        "model_id": model_id,
        "load_vino": load_vino,
        "load_tokenizer": load_tokenizer,  # Matches the ReloadQuery schema.
        "vino_file": vino_file,
    }
    try:
        response = requests.post(f"{BASE_SETTINGS.API_INTERNAL_URL}/reload", 
                                 json=payload,
                                 headers=headers)
        response.raise_for_status()
        # Return the success message from the API.
        return response.json()["message"]
    except Exception as e:
        return f"Failed to update configuration: {e}"

# Function to add a new label to the current list.
def add_label(current_labels, new_label):
    new_label = new_label.strip()
    # If a non-empty new label is provided and is not already in the list, add it.
    if new_label and new_label not in current_labels:
        current_labels.append(new_label)
    # Update the CheckboxGroup: update both available choices and selected values.
    updated_checkbox = gr.CheckboxGroup(choices=current_labels, value=current_labels)
    # Return the updated state and the updated CheckboxGroup.
    return current_labels, updated_checkbox

# Function to remove a label from the current list.
def remove_label(current_labels, label_to_remove):
    label_to_remove = label_to_remove.strip()
    if label_to_remove in current_labels:
        current_labels.remove(label_to_remove)
    # Update the CheckboxGroup accordingly.
    updated_checkbox = gr.CheckboxGroup(choices=current_labels, value=current_labels)
    return current_labels, updated_checkbox

# Function to process the text using the REST API.
def process_text(document_text, selected_labels, threshold):
    global BASE_SETTINGS
    headers = {
        "Content-Type":"application/json",
        "access_token": BASE_SETTINGS.API_KEY
        }
    # Build the payload in the format expected by your API.
    payload = {
        "documents": [{"text": document_text}],
        "labels": selected_labels,
        "threshold": threshold
    }
    
    try:
        response = requests.post(f"{BASE_SETTINGS.API_INTERNAL_URL}/process", 
                                 json=payload,
                                 headers=headers)
        response.raise_for_status()  # Raise error for bad responses.
        # The response has a dict itself
        result = response.json()['result'][0]
        # We need to add the labeltypes to the object so we cann add the color
        result['types'] = selected_labels
        processed_text = render_html(result)
        return processed_text
    except requests.RequestException as e:
        return f"Error calling the API: {e}"

# --- Build the Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# GLiNER-VINO Interface")
    
    gr.Markdown("## Update Model Configuration")
    
    with gr.Row():
        model_id = gr.Textbox(label="Model ID")
        load_vino = gr.Checkbox(label="Load Vino")
        load_tokenizer = gr.Checkbox(label="Load Tokenizer")
        vino_file = gr.Textbox(label="Vino File")
    
    with gr.Row():
        refresh_button = gr.Button("Load Current Config")
        update_button = gr.Button("Update Configuration")
    
    output = gr.Textbox(label="Response")
    
    # When the refresh button is clicked, fetch current config and fill the inputs.
    refresh_button.click(
        fetch_config,
        outputs=[model_id, load_vino, load_tokenizer, vino_file]
    )
    
    # When the update button is clicked, send the updated config.
    update_button.click(
        update_config_gradio,
        inputs=[model_id, load_vino, load_tokenizer, vino_file],
        outputs=output
    )
    gr.Markdown("## Input")
    gr.Markdown(
        "Paste your text below, select which labels to apply (modify the list as needed), "
        "and click **Process** to see the entities in the text. This model is multilingual."
    )
    
    # Text input for the document.
    text_input = gr.Textbox(
        lines=10,
        placeholder="Paste or type your text here...",
        label="Document Text"
    )
    
    # Panel for label management:
    with gr.Blocks():
        gr.Markdown("## Label Selection & Management")
        # Initialize state with some default labels.
        label_state = gr.State(["Person", "Organization", "Location"])
        
        # Checkbox group showing the current available labels.
        label_checkbox = gr.CheckboxGroup(
            choices=label_state.value,
            label="Select Labels",
            value=label_state.value,  # All labels selected by default.
            interactive=True
        )
        
        with gr.Row():
            add_label_input = gr.Textbox(
                lines=1,
                placeholder="Enter new label",
                label="Add Label"
            )
            add_label_button = gr.Button("Add Label")
            
            remove_label_input = gr.Textbox(
                lines=1,
                placeholder="Enter label to remove",
                label="Remove Label"
            )
            remove_label_button = gr.Button("Remove Label")
    
    # Include a slider for threshold if still needed.
    threshold_slider = gr.Slider(
        minimum=0, maximum=1, step=0.01, value=0.5, label="Threshold",
        info="Criteria for filtering candidates for an entity."
    )
    
    # Process button and output panel.
    process_button = gr.Button("Process")
    # Use an HTML component for output so that you can display colored/highlighted text.
    processed_output = gr.HTML(label="Processed Text (HTML)")
    
    # --- Set Up Callbacks ---
    # When Add Label button is clicked:
    add_label_button.click(
        fn=add_label,
        inputs=[label_state, add_label_input],
        outputs=[label_state, label_checkbox]
    )
    
    # When Remove Label button is clicked:
    remove_label_button.click(
        fn=remove_label,
        inputs=[label_state, remove_label_input],
        outputs=[label_state, label_checkbox]
    )
    
    # When the Process button is clicked:
    process_button.click(
        fn=process_text,
        inputs=[text_input, label_checkbox, threshold_slider],
        outputs=processed_output
    )

if __name__ == "__main__":
    # Launch Gradio so that it listens on all interfaces (0.0.0.0) on port 7860.
    demo.launch(server_name="0.0.0.0", server_port=7860)
