import gradio as gr

from functions import function_hw1, function_hw2, function_hw3, function_hw4, function_hw5
from UI import (
    create_demo_hw1,
    create_demo_hw2,
    create_demo_hw3,
    create_demo_hw4,
    create_demo_hw5,
)


def main():
    with gr.Blocks(title="Peking University Image Processing — Fall 2026") as demo:
        gr.Markdown("# Peking University Image Processing — Fall 2026")
        with gr.Tabs():
            with gr.TabItem("HW1: Color and Tone Adjustment"):
                create_demo_hw1(function_hw1)
            with gr.TabItem("HW2: DCGAN Face Generation and Editing"):
                create_demo_hw2(function_hw2)
            with gr.TabItem("HW3: Screen Replacement"):
                create_demo_hw3(function_hw3)
            with gr.TabItem("HW4: Sharpening and Denoising"):
                create_demo_hw4(function_hw4)
            with gr.TabItem("HW5: Histogram Equalization and Matching"):
                create_demo_hw5(function_hw5)

    demo.launch(server_name="127.0.0.1", server_port=8088, share=False)


if __name__ == "__main__":
    main()
