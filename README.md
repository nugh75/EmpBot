# EmpBot

This chatbot conducts a sentiment analysis related to the emotion of being moved.



## Requirements

- Python 3.10.12
- Packages listed in the `requirements.txt` file.
- Ollama configured locally to use GPT models.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/nugh75/chatbotc.git 
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv myenv
     ```

2.b activate the virtual enviroment

 ```bash
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
 ```

3. Install the dependencies:

  ```bash
    pip install -r requirements.txt
  ```



1. Run the Streamlit app:

    ```bash
    streamlit run appg.py
    ```
## Configuration

### Environment Variables

The `.env` file should contain the following environment variables:

- `API_KEY`: The API key for the ChatOpenAI model 


## Contributing

If you wish to contribute, please fork the repository and create a pull request with your changes.

## License

This project is licensed under the GPLv3 License. See the LICENSE file for more details.
