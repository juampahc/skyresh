import random
import html

def generate_random_color():
    """
    Generate a pastel color by mixing a random color with white.
    
    This is done by averaging a random number (0–255) with 255 (white) for
    each of the RGB channels. The result is a color that is always in the
    lighter half of the spectrum.
    """
    r = (random.randint(0, 255) + 255) // 2
    g = (random.randint(0, 255) + 255) // 2
    b = (random.randint(0, 255) + 255) // 2
    return f"#{r:02x}{g:02x}{b:02x}"

def render_html(data):
    """
    Render an HTML snippet from a dictionary containing text and entity annotations.
    
    The data dict is expected to have:
      - data["text"]: the full text string.
      - data["ents"]: a list of dicts, each with keys "start", "end", and "label".
      - data["types"]: a list of strings containing all posible labels
      
    Each unique entity label gets a randomly assigned background color.
    """
    text = data.get("text", "")
    entities = data.get("ents", [])
    
    # Sort entities by their starting index so we can iterate in order.
    entities = sorted(entities, key=lambda ent: ent["start"])
    
    # Map each entity label to a random color.
    label_to_color = {}
    for label in data['types']:
        if label not in label_to_color:
            label_to_color[label] = generate_random_color()
    
    # Build the HTML output.
    html_output = '<div style="line-height: 1.6; font-size: 16px;">\n'
    last_idx = 0
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["label"]
        color = label_to_color[label]
        
        # Append the text between the last entity and the current entity.
        html_output += html.escape(text[last_idx:start])
        
        # The text corresponding to the entity.
        entity_text = html.escape(text[start:end])
        
        # Create the mark element for the entity.
        html_output += (
            f'<mark class="entity" '
            f'style="display: inline-block; vertical-align: middle; background: {color}; '
            'padding: 0.3em 0.5em; margin: 0 0.2em; line-height: 1.2; border-radius: 0.35em;">'
            f'{entity_text}'
            f'<span style="display: inline-block; white-space: nowrap; font-size: 0.8em; '
            'font-weight: bold; margin-left: 0.5rem; line-height: 1;">'
            f'{label}</span>'
            f'</mark>'
        )
        last_idx = end

    # Append any text after the last entity.
    html_output += html.escape(text[last_idx:])
    html_output += "\n</div>"
    return html_output