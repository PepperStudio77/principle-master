# User Manual for `principle-master`

## Overview

`principle-master` is an AI-powered chatbot designed to help users build and refine their own principles to guide their
work and life. It provides tools for case reflection, principle creation, and content indexing.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd principle-master
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Commands

The application uses the `Click` library to provide a command-line interface. Below are the available commands:

1. **One time setup - Init the configuration of your LLM model. **:
   ```bash
   python main.py config-llm
   ```
    - This command initializes the configuration for the LLM model. It will prompt you to enter your OpenAI API key and
      other settings.
    - The configuration is saved in the `./principle-master/config/key.json` file.

2. **One time setup - Index Principle book's PDF version or other book serve you well as your guidelines **:
     ```bash
     python main.py index_contentr <pdf_path>  
     ```
    - `<pdf_path>`: Path to the PDF file to be indexed
    - The indexed content is stored in the `./principle-master/index` directory.

3. **Run Principle Master**:
     ```bash
     python main.py principle-master --verbose
     ```
    - `--verbose`: Enable verbose logging.
    - `--dynamic`: Use dynamic workflows for personalized principle creation. (Functionality is same, just another
      implementation for fun.)

---

## Features

### 1. **Case Reflection**

- Use the `CaseReflectionAgent` to reflect on cases based on Ray Dalio's principles.
- The agent will guide you through structured questions to help you analyze and learn from your experiences.

### 2. **Profile recording**
- The `ProfileAgent` allows you to record your profile, including strengths, weaknesses, and challenges.
- This information is used to tailor the chatbot's responses and recommendations.

### 3. **Advice**
- The `AdviceAgent` provides personalized advice based on your profile and existing case-reflection and book content you
  indexed.
- It also adjust your AI journal template accordingly based its advice for you.

### 3. **AI Journal**
- Create a daily journal and store under `principle-master/journals/` directory.

---

## Development

### Code Structure

- **`main.py`**: Entry point for the application.
- **`core/`**: Contains core logic for workflows, indexing, and case reflection.
- **`utils/`**: Utility functions, including LLM embedding.

### Adding New Commands

To add a new command, define a new `@click.command` function in `main.py` and register it using `consult.add_command()`.

---

## Troubleshooting

1. **Dependencies Not Installed**:
   Ensure you have activated the virtual environment and installed all dependencies:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Command Not Found**:
   Ensure you are running the commands from the project root directory.

3. **PDF Indexing Issues**:
   Verify the PDF file path and ensure it is accessible.

---

---

## Contact

For issues or feature requests, please open an issue in the GitHub repository.

--- 

This manual can be expanded further based on additional project details or user feedback.