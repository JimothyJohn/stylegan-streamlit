#!/usr/bin/env python3
import numpy as np
import streamlit as st
import stylegan
import utils

st.header("StyleGAN3 Playground")
st.sidebar.header("Operations")
program = st.sidebar.selectbox(
    "Choose a function", ["Generate", "Project", "Synthesize", "Align"], 0
)

st.header(program)

if program == "Generate":
    ElGenerator = stylegan.Generation()
    ElGenerator.GUI()

    # vector_batch = np.random.RandomState(user_seed).randn(1, Generator.z_dim)
    st.image(ElGenerator.GenerateImage())

elif program == "Project":
    projection = stylegan.Projection()
    run_projection = st.button("Project vectors")
    projection.GUI()
    if run_projection:
        projection.Project()

elif program == "Synthesize":
    mods = st.sidebar.multiselect(
        "Choose modifications", ["Modulate", "Mix", "Isolate"]
    )
    left, mid, right = st.columns(3)
    synthesis_one = stylegan.Synthesis(left, "left")
    synthesis_one.GUI()
    img = synthesis_one.Synthesize()

    if "Mix" in mods:
        synthesis_two = stylegan.Synthesis(mid, "mid")
        synthesis_three = stylegan.Synthesis(right, "right")
        synthesis_two.GUI()
        synthesis_three.GUI()
        synthesis_two.Synthesize()
        synthesis_three.Synthesize()
        mix_level = right.slider("Mix level", 1, 6, 4, 1) + 2
        img = synthesis_one.Mix(
            synthesis_two.mapping_batch,
            mix_level,
        )

        seed = int(abs(synthesis_one.mapping_batch[0][0][4] * 128))
        length = right.slider("Choose a length (seconds)", 0, 20, 4)
        video_name = right.text_input("Enter video name", "Mixed")
        if right.button("Save video"):
            st.header("Writing video...")
            stylegan.SaveVideo(
                synthesis_one.G,
                [
                    synthesis_one.mapping_batch,
                    synthesis_two.mapping_batch,
                    synthesis_three.mapping_batch,
                ],
                length,
            )

        st.header("Done!")

    if "Modulate" in mods:
        octave = right.slider("Octave:", 1, 10, 1, 1)
        loFrequency, hiFrequency = right.slider(
            "Choose a base frequency", 1, 100, (25, 75), 1
        )
        loFrequency /= 100000 * octave
        hiFrequency /= 100000 * octave
        right.write(f"Raw frequency output: {hiFrequency}")
        loMod, hiMod = right.slider(
            "Choose a modulation frequency", 1, 100, (25, 75), 1
        )
        loMod /= 100000 * octave
        hiMod /= 100000 * octave
        right.write(f"Raw modulation output: {hiMod}")
        # self.amplitude = st.slider('Choose an amplitude', 0., 4., 0., .1)
        amplitude = 1
        cutoff = right.slider("Choose a cutoff level", 1, 512, 1)
        cutoff_dir = right.checkbox("Invert cutoff")
        sinepolarity = right.checkbox("Sin/Cos")
        waveform = right.selectbox("Choose a waveform", ["ramp", "sine"], 1)

        img = synthesis_one.ModulateMapping(
            hiFrequency,
        )

        img = synthesis_one.ModulateMapping(
            hiMod,
        )

        if right.button("Save video"):
            stylegan.SaveVideo(
                synthesis_one,
                0.5,
                [loFrequency, hiFrequency],
                [loMod, hiMod],
            )

    if "Isolate" in mods:
        iso_channel = mid.slider("Isolate channel", 1, 16, 8, 1) - 1

    if len(mods) > 0:
        st.write(f"First 5: {img[0][0][:5]}")
        st.image(img)

elif program == "Align":
    left, right = st.columns(2)

    image_file = left.selectbox("Choose an image: ", stylegan.image_list, 0)
    run_alignment = left.button("Align image")

    right.image(f"out/{image_file}")
    if run_alignment:
        aligned_face = np.asarray(utils.AlignFace(f"{image_file}"))
        right.image(aligned_face)
