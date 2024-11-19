This project classifies plant diseases from uploaded images using a custom Enhanced ResNet18 model. The web application is implemented using Streamlit and allows users to upload an image, which the model processes to predict the disease class. The application supports 15 plant disease categories and is designed for real-time inference.

Installation and Running Instructions
Prerequisites
Python 3.8 or above
CUDA (optional for GPU support)
Required Python packages (listed in requirements.txt)

Clone the Repository
git clone <https://github.com/balanarch/finaleldar>

Install Dependencies Ensure you have all required Python packages by running:
pip install -r requirements.txt

Prepare the Dataset

Place your dataset in a folder named Dataset in the root directory.
The dataset should follow the structure:

Download the Pretrained Model
Download the model weights file (model.pth) and place it in the root directory.

Run the Application Start the Streamlit server:
streamlit run app.py

Special Requirements
Pretrained Weights: Ensure the file model.pth is present in the root directory. You can either download it or train the model yourself using the training script.
PyTorch: Install PyTorch with the appropriate version for your system. Use PyTorch's installation page to select the correct command based on your CUDA availability.
Image Size: Uploaded images will be resized to 128x128 pixels during preprocessing

Contributing
Fork the repository.
Create a feature branch:
git checkout -b feature-name
Commit your changes:
git commit -m "Add your message"
Push to the branch:
git push origin feature-name
Open a Pull Request.

For questions or contributions, feel free to reach out at:

Email: eoralgaziev04@gmail.com
GitHub: https://github.com/balanarch