# pip install streamlit
# pip install pytorch-lightning
# streamlit run app.py
import numpy as np
import os
import json
# import requests # Removed as not used in the provided code
from io import BytesIO
# from stqdm import stqdm # Removed as not used in the provided code
from PIL import Image
import streamlit as st
import torch
import random
import torchvision.utils as vutils # Can be useful, but sticking to PIL for saving consistency
import torchvision.transforms as transforms
import torch.nn.functional as F

# --- Import your model definitions ---
# Ensure these files are in the same directory or accessible via PYTHONPATH
try:
    import modelf as modelf # Should contain MicrographBigGAN and Generator classes
    import model2 as model2 # Should contain Generator class for the other model
    import cnn_model as cnn_model # Should contain MicrostructureCNN class
    import resnet_classify as resnet_classify # Should contain MicrostructureResNet class
except ImportError as e:
    st.error(f"Fatal Error: Could not import necessary model definition file(s). Please ensure modelf.py, model2.py, cnn_model.py, and resnet_classify.py are accessible. Details: {e}")
    st.stop() # Stop the app if essential files are missing

# --- Configuration ---
# Update these paths if your files are located elsewhere
GAN_CHECKPOINT_WITH_CONSTITUENT = 'last-v1.ckpt'
GAN_CHECKPOINT_WITHOUT_CONSTITUENT = 'last.pth'
CNN_CHECKPOINT = 'microstructure_model.pth'
RESNET_CHECKPOINT = 'resnet18_microstructure.pth'

# --- Helper Classes/Functions ---

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# --- Model Loading Functions ---

# Use allow_output_mutation=True because model objects are mutable
# Use suppress_st_warning=True to avoid warnings about hashing mutable objects if needed
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_gan_model(with_microconstituent):
    """Loads the appropriate GAN generator model."""
    if with_microconstituent:
        st.write(f"Loading Generator model (from {GAN_CHECKPOINT_WITH_CONSTITUENT})...")
        if not os.path.exists(GAN_CHECKPOINT_WITH_CONSTITUENT):
             st.error(f"Error: Checkpoint file '{GAN_CHECKPOINT_WITH_CONSTITUENT}' not found.")
             return None
        try:
            # --- Load the full LightningModule checkpoint ---
            gan_model = modelf.MicrographBigGAN.load_from_checkpoint(
                GAN_CHECKPOINT_WITH_CONSTITUENT,
                map_location='cpu'
            )
            # --- Extract the generator component ---
            model = gan_model.generator
            model.eval() # Set to evaluation mode
            st.success("Generator model (with microconstituent) loaded.")
            return model

        except AttributeError as e:
             st.error(f"Error: Could not find 'MicrographBigGAN' class in 'modelf.py' or 'generator' attribute in the loaded checkpoint '{GAN_CHECKPOINT_WITH_CONSTITUENT}'. Details: {e}")
             return None
        except Exception as e:
            st.error(f"Error loading model from '{GAN_CHECKPOINT_WITH_CONSTITUENT}': {e}")
            return None

    else: # Load the model without microconstituent
        st.write(f"Loading Generator model (from {GAN_CHECKPOINT_WITHOUT_CONSTITUENT})...")
        if not os.path.exists(GAN_CHECKPOINT_WITHOUT_CONSTITUENT):
             st.error(f"Error: Model file '{GAN_CHECKPOINT_WITHOUT_CONSTITUENT}' not found.")
             return None
        try:
            # Assuming this checkpoint *only* contains the state_dict for model2.Generator
            model = model2.Generator()
            state_dict = torch.load(GAN_CHECKPOINT_WITHOUT_CONSTITUENT, map_location=torch.device('cpu'))
            # Handle cases where the checkpoint might be a dictionary (e.g., saved by Lightning)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                 model.load_state_dict(state_dict['state_dict'])
            else:
                 model.load_state_dict(state_dict)

            model.eval() # Set to evaluation mode
            st.success("Generator model (without microconstituent) loaded.")
            return model
        except Exception as e:
            st.error(f"Error loading model from '{GAN_CHECKPOINT_WITHOUT_CONSTITUENT}': {e}")
            return None

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_cnn_model():
    """Loads the CNN classification model."""
    st.write(f"Loading CNN classification model (from {CNN_CHECKPOINT})...")
    if not os.path.exists(CNN_CHECKPOINT):
        st.error(f"Error: Checkpoint file '{CNN_CHECKPOINT}' not found.")
        return None
    try:
        # model = cnn_model.MicrostructureCNN(num_classes=6) # Assuming 6 classes
        # checkpoint = torch.load(CNN_CHECKPOINT, map_location=torch.device("cpu"))

        model = cnn_model.MicrostructureCNN(num_classes=6)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Load pretrained weights
        checkpoint = torch.load(CNN_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Handle different checkpoint saving formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
             model.load_state_dict(checkpoint['state_dict'])
        else:
             model.load_state_dict(checkpoint)
        model.eval()
        st.success("CNN classification model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading CNN model from '{CNN_CHECKPOINT}': {e}")
        return None


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_resnet_model():
    """Loads the ResNet classification model."""
    st.write(f"Loading ResNet classification model (from {RESNET_CHECKPOINT})...")
    if not os.path.exists(RESNET_CHECKPOINT):
        st.error(f"Error: Checkpoint file '{RESNET_CHECKPOINT}' not found.")
        return None
    try:
        # model = resnet_classify.MicrostructureResNet(num_classes=6) # Assuming 6 classes
        # checkpoint = torch.load(RESNET_CHECKPOINT, map_location=torch.device("cpu"))

        
        model = resnet_classify.MicrostructureResNet(num_classes=6)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Load pretrained weights
        checkpoint = torch.load(RESNET_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
         # Handle different checkpoint saving formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
             model.load_state_dict(checkpoint['state_dict'])
        else:
             model.load_state_dict(checkpoint)
        model.eval()
        st.success("ResNet classification model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading ResNet model from '{RESNET_CHECKPOINT}': {e}")
        return None


# --- Image Generation Functions ---

# Caching this function means if the exact same inputs are provided again,
# it will return the previously generated image instantly.
# @st.cache # Caching generation can be good, but remove if debugging model output changes
def generate_img(model, noise, y_temp, y_time, y_cool):
    """Generates image using the model WITHOUT microconstituent input."""
    if model is None: return None
    model.eval() # Ensure eval mode

    # Convert conditions to tensors
    y_temp = torch.tensor([y_temp], dtype=torch.long)
    y_time = torch.tensor([y_time], dtype=torch.long)
    y_cool = torch.tensor([y_cool], dtype=torch.long)
    # noise is assumed to be a tensor already

    with torch.no_grad():
        synthetic = model(noise, y_temp, y_time, y_cool)
        # Handle potential nested output
        if isinstance(synthetic, (list, tuple)):
             synthetic = synthetic[0]
        # Denormalize: [-1, 1] -> [0, 1]
        synthetic = 0.5 * synthetic + 0.5
        synthetic = torch.clamp(synthetic, 0, 1)

    # Convert to NumPy array for display: [B, C, H, W] -> [H, W, C]
    numpy_image = synthetic.squeeze(0).cpu().numpy() # Squeeze batch dimension
    # Transpose channel axis: (C, H, W) -> (H, W, C)
    return np.transpose(numpy_image, (1, 2, 0))

# @st.cache # Caching generation can be good, but remove if debugging model output changes
def generate_img_with_constituent(model, noise, y_temp, y_time, y_cool, y_constituent):
    """Generates image using the model WITH microconstituent input."""
    if model is None: return None
    model.eval() # Ensure eval mode

    # Convert conditions to tensors
    y_temp = torch.tensor([y_temp], dtype=torch.long)
    y_time = torch.tensor([y_time], dtype=torch.long)
    y_cool = torch.tensor([y_cool], dtype=torch.long)
    y_constituent = torch.tensor([y_constituent], dtype=torch.long)
    # noise is assumed to be a tensor already

    with torch.no_grad():
        synthetic = model(noise, y_temp, y_time, y_cool, y_constituent)
         # Handle potential nested output
        if isinstance(synthetic, (list, tuple)):
             synthetic = synthetic[0]
        # Denormalize: [-1, 1] -> [0, 1]
        synthetic = 0.5 * synthetic + 0.5
        synthetic = torch.clamp(synthetic, 0, 1)

    # Convert to NumPy array for display: [B, C, H, W] -> [H, W, C]
    numpy_image = synthetic.squeeze(0).cpu().numpy() # Squeeze batch dimension
    # Transpose channel axis: (C, H, W) -> (H, W, C)
    return np.transpose(numpy_image, (1, 2, 0))


# --- Streamlit Page Functions ---

def generate_image_page(with_microconstituent=False):
    """Handles the UI and logic for GAN image generation."""
    st.sidebar.title('Processing Conditions')

    # --- Define Mappings ---
    # Ensure these match the training configuration *exactly*
    temp_map = {700: 5, 750: 6, 800: 1, 900: 2, 970: 0, 1000: 4, 1100: 3}
    time_map = {'5M': 3, '90M': 0, '1H': 6, '3H': 2, '8H': 4, '24H': 1, '48H': 7, '85H': 5}
    cool_map = {'Quench': 0, 'Air Cool': 1, 'Furnace Cool': 2, '650C-1H': 3} # Check exact naming 'Air Cool' vs 'AR', 'Furnace Cool' vs 'FC' etc.
    constituent_map = {
        'spheroidite': 0,
        'pearlite+spheroidite': 1, # Or 3? Double check mapping
        'pearlite': 4, # Or 2? Double check mapping
        'spheroidite+widmanstatten': 2, # Or 3? Double check mapping
        'network': 1, # Or 4? Double check mapping
        'pearlite+widmanstatten': 5
    }
    # Create reverse maps for display/filename if needed
    # rev_constituent_map = {v: k for k, v in constituent_map.items()} # Example

    # --- Get User Input ---
    # Use the keys from the maps for the selectbox options
    anneal_temp_display = st.sidebar.selectbox('Annealing Temperature °C', list(temp_map.keys()))
    anneal_time_display = st.sidebar.selectbox('Annealing Time (M: Minutes, H: Hours)', list(time_map.keys()))
    cooling_display = st.sidebar.selectbox('Cooling Type', list(cool_map.keys()))

    constituent_display = None
    if with_microconstituent:
        constituent_display = st.sidebar.selectbox('Microconstituent Type', list(constituent_map.keys()))

    # --- Load Model ---
    model = load_gan_model(with_microconstituent)
    if model is None:
        # Error message already shown by load_gan_model
        st.warning("Cannot proceed with generation as model failed to load.")
        return # Stop if model loading failed

    # --- Latent Vector (Noise) ---
    st.sidebar.subheader('Generate Latent Vector (Noise)')
    # Use session state to keep the seed consistent unless button is pressed
    if 'gan_seed' not in st.session_state:
         st.session_state['gan_seed'] = 7 # Default seed

    if st.sidebar.button('Generate New Noise'):
        st.session_state['gan_seed'] = random.randint(0, 10000)
    seed = st.session_state['gan_seed']
    st.sidebar.text(f"Current Seed: {seed}")

    # Generate noise using the selected seed
    rng = np.random.RandomState(seed)
    # Ensure noise shape matches model input (e.g., B=1, Z_DIM=384)
    # Check Z_DIM required by your specific Generator definition
    z_dim = 384 # Assuming Z dimension is 384 based on previous script
    noise_np = rng.normal(0, 1, (1, z_dim)).astype(np.float32)
    noise_tensor = torch.tensor(noise_np) # Convert to tensor for the model

    # --- Get Condition Indices ---
    y_temp = temp_map[anneal_temp_display]
    y_time = time_map[anneal_time_display]
    y_cool = cool_map[cooling_display]

    # --- Generate Image ---
    st.subheader('Generated Microstructure')
    image_out_np = None
    with st.spinner('Generating image...'):
        if with_microconstituent:
            if constituent_display is None:
                 st.error("Microconstituent selection is required.")
                 return
            y_constituent = constituent_map[constituent_display]
            image_out_np = generate_img_with_constituent(model, noise_tensor, y_temp, y_time, y_cool, y_constituent)
        else:
            image_out_np = generate_img(model, noise_tensor, y_temp, y_time, y_cool)

    # --- Display and Download ---
    if image_out_np is not None:
        st.text(f"Generated with Seed: {seed}")
        # Display the NumPy array (H, W, C), values should be [0, 1]
        st.image(image_out_np, caption="Generated Micrograph", use_column_width=False, clamp=True)

        # --- Prepare for Download ---
        # Convert numpy [0,1] float to [0,255] uint8 for saving as standard PNG
        try:
            # Handle grayscale (if C=1) vs color
            if image_out_np.shape[-1] == 1:
                pil_image_out = image_out_np.squeeze(-1) # Remove channel dim -> (H, W)
            else:
                 pil_image_out = image_out_np # Assume (H, W, C) for color

            # Scale to 0-255 and convert to uint8
            pil_image_out_uint8 = (pil_image_out * 255).astype(np.uint8)

            im_pil = Image.fromarray(pil_image_out_uint8)
            buf = BytesIO()
            im_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            # Construct filename
            filename_parts = [anneal_temp_display, anneal_time_display, cooling_display]
            if with_microconstituent and constituent_display:
                 filename_parts.append(constituent_display.replace('+','_')) # Replace '+' for filesystem compatibility
            filename_parts.append(f"seed{seed}")
            file_name = f"{'-'.join(map(str, filename_parts))}.png"

            st.download_button(
                label="Download Micrograph",
                data=byte_im,
                file_name=file_name,
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Error preparing image for download: {e}")
            st.error(f"Generated image shape: {image_out_np.shape}, dtype: {image_out_np.dtype}")

    else:
        st.error("Image generation failed. Check model loading and parameters.")


def cnn_page_template(model_type, model_load_func):
    """Template function for micrograph classification pages."""
    st.title(f"Micrograph Classification ({model_type})")

    csv_file = 'with_microconstituent/input\highcarbon-micrographs/new_metadata.xlsx'
    img_dir = 'with_microconstituent/input/highcarbon-micrographs/For Training/Cropped'
    transform = transforms.Compose([
            transforms.Resize((256, 256)), # Resize to expected input size
            transforms.ToTensor(), # Converts to [0, 1] tensor [C, H, W]
            transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] (assuming single channel)
        ])
    dataset = cnn_model.MicrographDataset(csv_file=csv_file, img_dir=img_dir, transform=transform, return_paths=False)
    label_to_idx = dataset.label_to_idx
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    uploaded_file = st.file_uploader("Upload a micrograph image", type=["png", "jpg", "jpeg", "tif"], key=f"uploader_{model_type}")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('L') # Convert to grayscale
            st.image(image, caption="Uploaded Micrograph", use_column_width=True)
        except Exception as e:
            st.error(f"Error opening or processing uploaded image: {e}")
            return

        # --- Preprocessing ---
        # Ensure this matches the preprocessing used during classifier training
       
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            img_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension [B, C, H, W]
        except Exception as e:
            st.error(f"Error transforming image for model: {e}")
            return

        # --- Load Model ---
        model = model_load_func()
        if model is None:
            st.error("Classifier model failed to load. Cannot perform classification.")
            return

        # --- Prediction ---
        try:
            with st.spinner(f'Classifying using {model_type}...'), torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                predicted_class_idx = torch.argmax(probs, dim=1).item()
                predicted_label = idx_to_label[predicted_class_idx]
                print(f"📷 Predicted Class: {predicted_label} ({predicted_class_idx})")
                print(f"Class Probabilities: {F.softmax(output, dim=1)*100}")

                 # Define class names in correct order matching model training
                class_names = [
                        'network',
                        'pearlite',
                        'pearlite+spheroidite',
                        'pearlite+widmanstatten',
                        'spheroidite',
                        'spheroidite+widmanstatten'
                    ]

            if 0 <= predicted_class_idx < len(class_names):
                    predicted_class_name = class_names[predicted_class_idx]
                    st.success(f"🔍 Predicted Microconstituent: **{predicted_class_name}**")

                    # Display class probabilities
                    st.subheader("📊 Class Probabilities:")
                    probs_list = probs[0].cpu().numpy()  # Convert to numpy array
                    prob_dict = {class_names[i]: probs_list[i] for i in range(len(class_names))}
                    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

                    for class_name, prob in sorted_probs.items():
                        st.write(f"- **{class_name}**: {prob:.2%}")

            else:
                    st.error(f"⚠️ Model predicted an invalid class index: {predicted_class_idx}")

        except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

def resnet_page_template(model_type, model_load_func):
    """Template function for micrograph classification pages."""
    st.title(f"Micrograph Classification ({model_type})")

    csv_file = 'with_microconstituent/input\highcarbon-micrographs/new_metadata.xlsx'
    img_dir = 'with_microconstituent/input/highcarbon-micrographs/For Training/Cropped'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = resnet_classify.MicrographDataset(csv_file, img_dir, transform=None)
    idx_to_label = {idx: label for label, idx in dataset.label_to_idx.items()}
    

    uploaded_file = st.file_uploader("Upload a micrograph image", type=["png", "jpg", "jpeg", "tif"], key=f"uploader_{model_type}")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('L') # Convert to grayscale
            st.image(image, caption="Uploaded Micrograph", use_column_width=True)
        except Exception as e:
            st.error(f"Error opening or processing uploaded image: {e}")
            return

        # --- Preprocessing ---
        # Ensure this matches the preprocessing used during classifier training
       
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            img_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension [B, C, H, W]
        except Exception as e:
            st.error(f"Error transforming image for model: {e}")
            return

        # --- Load Model ---
        model = model_load_func()
        if model is None:
            st.error("Classifier model failed to load. Cannot perform classification.")
            return

        # --- Prediction ---
        try:
            with st.spinner(f'Classifying using {model_type}...'), torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                predicted_class_idx = torch.argmax(probs, dim=1).item()
                predicted_label = idx_to_label[predicted_class_idx]
                print(f"📷 Predicted Class: {predicted_label} ({predicted_class_idx})")
                print(f"Class Probabilities: {F.softmax(output, dim=1)*100}")

                 # Define class names in correct order matching model training
                class_names = [
                        'network',
                        'pearlite',
                        'pearlite+spheroidite',
                        'pearlite+widmanstatten',
                        'spheroidite',
                        'spheroidite+widmanstatten'
                    ]

            if 0 <= predicted_class_idx < len(class_names):
                    predicted_class_name = class_names[predicted_class_idx]
                    st.success(f"🔍 Predicted Microconstituent: **{predicted_class_name}**")

                    # Display class probabilities
                    st.subheader("📊 Class Probabilities:")
                    probs_list = probs[0].cpu().numpy()  # Convert to numpy array
                    prob_dict = {class_names[i]: probs_list[i] for i in range(len(class_names))}
                    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

                    for class_name, prob in sorted_probs.items():
                        st.write(f"- **{class_name}**: {prob:.2%}")

            else:
                    st.error(f"⚠️ Model predicted an invalid class index: {predicted_class_idx}")

        except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
       

# --- Main Application Logic ---

def main():
    st.set_page_config(layout="wide") # Optional: Use wider layout
    st.title("Microstructure GAN and Classification Application")

    # --- Main Option Selection ---
    option = st.selectbox("Select Task", [
        "Create image with microconstituent (GAN)",
        "Create image without microconstituent (GAN)",
        "Classify an image (CNN)",
        "Classify an image (ResNet)"
    ], key="main_task_selector")

    # --- Routing ---
    if option == "Create image without microconstituent (GAN)":
        generate_image_page(with_microconstituent=False)

    elif option == "Create image with microconstituent (GAN)":
        generate_image_page(with_microconstituent=True)

    elif option == "Classify an image (CNN)":
        cnn_page_template(model_type="CNN", model_load_func=load_cnn_model)

    elif option == "Classify an image (ResNet)":
        resnet_page_template(model_type="ResNet", model_load_func=load_resnet_model)

# --- App Entry Point ---
if __name__ == "__main__":
    main()
    # Optional Footer
    st.markdown('---')
    st.markdown('<div style="text-align: center; color: grey;">LBP PROJECT (Dakshesh)</div>', unsafe_allow_html=True)
    
    
# .\venv\Scripts\activate    
# streamlit run app_new.py
    
    