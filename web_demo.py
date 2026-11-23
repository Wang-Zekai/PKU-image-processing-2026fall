from functions import function_hw5
from UI import create_demo_hw5


def main():
    demo = create_demo_hw5(function_hw5)
    demo.launch(server_name="127.0.0.1", server_port=8088, share=False)


if __name__ == "__main__":
    main()
