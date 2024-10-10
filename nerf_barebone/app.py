import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from nerf_vis import (
    NeRF,
    PositionalEncoder,
    render_image,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.realpath(__file__))
data = np.load(os.path.join(script_dir,"tiny_nerf_data.npz"))


# ---------------------------
# Streamlit App
# ---------------------------

def main():
    st.title("NeRF Interactive Renderer")
    st.write("Adjust the **Hue** and **Saturation** sliders to see real-time rendering.")

    # Sidebar for sliders
    st.sidebar.header("Adjust Environmental Parameters")

    hue = st.sidebar.slider(
        "Hue",
        min_value=-30.0,
        max_value=30.0,
        value=0.0,
        step=1.0,
    )

    saturation = st.sidebar.slider(
        "Saturation",
        min_value=-0.5,
        max_value=0.5,
        value=0.0,
        step=0.05,
    )

    # Load models once
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading NeRF models..."):
            script_dir = os.path.dirname(os.path.realpath(__file__))

            # Initialize encoders
            d_input = 3
            n_freqs = 10
            log_space = True
            encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space).to(device)
            encode = lambda x: encoder(x)

            encoder_env = PositionalEncoder(d_input=2, n_freqs=n_freqs, log_space=log_space).to(device)
            encode_env = lambda x: encoder_env(x)

            n_freqs_views = 4
            encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views, log_space=log_space).to(device)
            encode_viewdirs = lambda x: encoder_viewdirs(x)
            d_viewdirs = encoder_viewdirs.d_output

            # Initialize models
            n_layers = 2
            d_filter = 128
            skip = []
            model = NeRF(
                d_input=encoder.d_output + encoder_env.d_output,
                n_layers=n_layers,
                d_filter=d_filter,
                skip=skip,
                d_viewdirs=d_viewdirs,
            ).to(device)

            fine_model = NeRF(
                d_input=encoder.d_output + encoder_env.d_output,
                n_layers=n_layers,
                d_filter=d_filter,
                skip=skip,
                d_viewdirs=d_viewdirs,
            ).to(device)

            # Load pretrained weights
            coarse_model_path = os.path.join(script_dir, "nerf.pt")
            fine_model_path = os.path.join(script_dir, "nerf-fine.pt")

            if os.path.exists(coarse_model_path):
                model.load_state_dict(torch.load(coarse_model_path, map_location=device))
                model.eval()
                st.sidebar.success("Loaded coarse NeRF model.")
            else:
                st.sidebar.error("Coarse NeRF model not found.")

            if os.path.exists(fine_model_path):
                fine_model.load_state_dict(torch.load(fine_model_path, map_location=device))
                fine_model.eval()
                st.sidebar.success("Loaded fine NeRF model.")
            else:
                fine_model = None
                st.sidebar.warning("Fine NeRF model not found. Proceeding without fine model.")

            # Load data path
            data_path = script_dir  # Assuming 'tiny_nerf_data.npz' is in the same directory

            st.session_state['encoder'] = encoder
            st.session_state['encode'] = encode
            st.session_state['encoder_env'] = encoder_env
            st.session_state['encode_env'] = encode_env
            st.session_state['encoder_viewdirs'] = encoder_viewdirs
            st.session_state['encode_viewdirs'] = encode_viewdirs
            st.session_state['model'] = model
            st.session_state['fine_model'] = fine_model
            st.session_state['data_path'] = data_path
            st.session_state['models_loaded'] = True

    if 'models_loaded' in st.session_state:
        encoder = st.session_state['encoder']
        encode = st.session_state['encode']
        encoder_env = st.session_state['encoder_env']
        encode_env = st.session_state['encode_env']
        encoder_viewdirs = st.session_state['encoder_viewdirs']
        encode_viewdirs = st.session_state['encode_viewdirs']
        model = st.session_state['model']
        fine_model = st.session_state['fine_model']
        data_path = st.session_state['data_path']

        # Render the image
        with st.spinner("Rendering... This may take a few seconds."):
            rendered_image, ground_truth = render_image(
                model=model,
                fine_model=fine_model,
                encode=encode,
                encode_env=encode_env,
                encode_viewdirs=encode_viewdirs,
                test_pose=torch.Tensor(data["poses"][13]).to(device),  # Using idx=13 as in your code
                hue=hue,
                saturation=saturation,
                data_path=data_path,
                testimgidx=13,
                near=2.0,
                far=6.0,
                n_samples=64,
                perturb=True,
                inverse_depth=False,
                n_samples_hierarchical=64,
                perturb_hierarchical=True,
                chunksize=2**14,
            )

        # Display images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.header("Rendered Image")
            st.image(rendered_image, channels="RGB")

        with col2:
            st.header("Ground Truth Image")
            st.image(ground_truth, channels="RGB")

if __name__ == "__main__":
    main()