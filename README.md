# LLM Comparison Toolkit

A Python tool for comparing responses from multiple Large Language Model APIs side by side.

<!-- This is a dummy comment added for verification purposes -->

## Features

- Compares responses from three AI APIs:
  - Inception Labs API (mercury, gpt-3.5-turbo, gpt-4, etc.)
  - OpenAI API (gpt-3.5-turbo, gpt-4, gpt-4o, etc.)
  - Google Gemini AI (gemini-pro, gemini-1.5-pro, etc.)
- **Two comparison modes:**
  - **Single Prompt**: Compare responses to one prompt across all APIs
  - **Prompt Chains**: Execute multi-step chains where output feeds into next step
- **Configurable model selection** - Choose specific models for each provider
- Measures and records response times for each API and chain step
- Saves results in CSV format for easy analysis
- Reads prompts from external text files or JSON chain definitions
- Handles API errors gracefully
- Command-line options to list available models

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Step-by-Step Installation

1. **Clone or download this repository:**
   ```bash
   git clone <repository-url>
   cd LLM-comparison-toolkit
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env file with your actual API keys
   nano .env  # or use any text editor
   ```

5. **Get your API Keys:**
   - **OpenAI**:
     - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
     - Create an account and generate an API key
     - Add to `.env` as: `OPENAI_API_KEY=sk-your-key-here`

   - **Google Gemini**:
     - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
     - Create a Google account and generate an API key
     - Add to `.env` as: `GEMINI_API_KEY=your-key-here`

   - **Inception Labs**:
     - Contact Inception Labs for API access
     - Add to `.env` as: `INCEPTION_API_KEY=your-key-here`
     - Note: You may need to adjust the API endpoint in the code

## How to Run

### Basic Usage

#### Single Prompt Comparison

1. **Prepare your prompt:**
   ```bash
   # Edit the prompt file with your question
   echo "What are the benefits of renewable energy?" > prompt.txt
   ```

2. **Run the comparison:**
   ```bash
   python llm_comparison.py
   ```

#### Prompt Chain Comparison

1. **Prepare your chain:**
   ```bash
   # Use the provided example or create your own
   cp simple_chain_example.json input_prompt_chain.json

   # Or edit the chain file
   nano input_prompt_chain.json
   ```

2. **Run the chain comparison:**
   ```bash
   python llm_comparison.py --chain
   ```

#### Configuration (Optional)

```bash
# View available models
python llm_comparison.py --models

# Edit models_config.json to select different models
nano models_config.json
```

#### Results

- Console output shows a summary
- **Single prompt**: Results saved to `llm_comparison_results.csv`
- **Chain**: Results saved to `llm_chain_results.csv`

### Example Run

```bash
$ python llm_comparison.py
Starting LLM Comparison...
============================================================
CURRENT MODEL CONFIGURATION
============================================================
Inception Labs: mercury
OpenAI: gpt-3.5-turbo
Gemini: gemini-pro
Max Tokens: 1000
Temperature: 0.7
============================================================

Prompt: What are the benefits of renewable energy?

Calling APIs...
- Calling Inception Labs API (mercury)...
- Calling OpenAI API (gpt-3.5-turbo)...
- Calling Gemini AI (gemini-pro)...

Results saved to: llm_comparison_results.csv

================================================================================
COMPARISON SUMMARY
================================================================================
Prompt: What are the benefits of renewable energy?

Inception Labs - mercury (2.341s):
  Success: True
  Response: Renewable energy offers numerous benefits including reduced carbon emissions, energy independence, job creation...

OpenAI - gpt-3.5-turbo (1.823s):
  Success: True
  Response: Renewable energy sources like solar, wind, and hydroelectric power provide several key advantages...

Gemini AI - gemini-pro (1.456s):
  Success: True
  Response: The transition to renewable energy brings multiple benefits for both the environment and economy...
```

### Command Line Options

```bash
# Run single prompt comparison
python llm_comparison.py

# Run prompt chain comparison
python llm_comparison.py --chain

# List all available models
python llm_comparison.py --models

# Show help
python llm_comparison.py --help
```

## Model Configuration

### Configuration File Structure

The `models_config.json` file controls which models are used for comparison:

```json
{
  "inception_labs": {
    "available_models": ["mercury", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    "default_model": "mercury"
  },
  "openai": {
    "available_models": ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"],
    "default_model": "gpt-3.5-turbo"
  },
  "gemini": {
    "available_models": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
    "default_model": "gemini-pro"
  },
  "selected_models": {
    "inception_labs": "mercury",
    "openai": "gpt-3.5-turbo",
    "gemini": "gemini-pro"
  },
  "api_parameters": {
    "max_tokens": 1000,
    "temperature": 0.7,
    "timeout_seconds": 60
  }
}
```

### Changing Models

1. **View available models:**
   ```bash
   python llm_comparison.py --models
   ```

2. **Edit configuration:**
   ```bash
   nano models_config.json
   ```

3. **Update the `selected_models` section:**
   ```json
   "selected_models": {
     "inception_labs": "gpt-4",
     "openai": "gpt-4o",
     "gemini": "gemini-1.5-pro"
   }
   ```

### Model Recommendations

- **For speed**: Use `gpt-3.5-turbo`, `gemini-1.5-flash`, `mercury`
- **For quality**: Use `gpt-4o`, `gemini-1.5-pro`, `gpt-4`
- **For cost-effectiveness**: Use `gpt-3.5-turbo`, `gemini-pro`, `mercury`

## Prompt Chains

### What are Prompt Chains?

Prompt chains allow you to execute a sequence of LLM calls where the output of one step becomes the input for the next step. This enables complex workflows like:

- **Research → Analysis → Summary → Action Items**
- **Draft → Review → Improve → Finalize**
- **Question → Research → Synthesize → Recommend**

### Chain File Structure

Create a JSON file with the following structure:

```json
{
  "chain_name": "Your Chain Name",
  "description": "Description of what this chain does",
  "steps": [
    {
      "step_id": 1,
      "name": "Step Name",
      "prompt": "Your prompt here",
      "use_previous_output": false,
      "output_variable": "variable_name"
    },
    {
      "step_id": 2,
      "name": "Next Step",
      "prompt": "Process this data: {previous_output}",
      "use_previous_output": true,
      "output_variable": "next_variable"
    }
  ],
  "chain_parameters": {
    "max_tokens_per_step": 1500,
    "temperature": 0.7,
    "timeout_seconds": 60,
    "continue_on_error": false,
    "save_intermediate_results": true
  }
}
```

### Chain Parameters

- **`use_previous_output`**: If `true`, replaces `{previous_output}` in the prompt with the previous step's response
- **`continue_on_error`**: If `true`, continues executing remaining steps even if one fails
- **`max_tokens_per_step`**: Token limit for each step (can override global setting)
- **`save_intermediate_results`**: Whether to save each step's output

### Example Chains

The toolkit includes example chains:
- **`simple_chain_example.json`**: Basic 2-step writing improvement chain
- **`input_prompt_chain.json`**: Complex 4-step research and analysis chain

### Chain Output

Chain results include:
- **Step-by-step execution details**
- **Timing for each step and total time**
- **Success/failure status for each step**
- **Final output from the last successful step**
- **Detailed CSV with all intermediate results**

## Understanding the Results

### Console Output
The script provides real-time feedback and a summary including:
- **Response Time**: How long each API took to respond (in seconds)
- **Success Status**: Whether the API call was successful
- **Response Preview**: First 200 characters of each response

### CSV File Structure
The `llm_comparison_results.csv` file contains two rows:

**Row 1 - Main Results:**
| Column | Description |
|--------|-------------|
| Timestamp | When the comparison was run |
| Prompt | The exact prompt sent to all APIs |
| Inception_Response | Full response from Inception Labs |
| Inception_Time_Seconds | Response time in seconds |
| Inception_Success | True/False for API success |
| OpenAI_Response | Full response from OpenAI |
| OpenAI_Time_Seconds | Response time in seconds |
| OpenAI_Success | True/False for API success |
| Gemini_Response | Full response from Gemini |
| Gemini_Time_Seconds | Response time in seconds |
| Gemini_Success | True/False for API success |

**Row 2 - Timing Summary:**
A quick reference row showing just the response times for easy comparison.

### Analyzing Results

1. **Response Quality**: Compare the actual responses to see which API provides the most helpful, accurate, or detailed answer for your use case.

2. **Speed Comparison**: Check the timing columns to see which API responds fastest. Typical ranges:
   - **Fast**: < 2 seconds
   - **Moderate**: 2-5 seconds
   - **Slow**: > 5 seconds

3. **Reliability**: Check the Success columns. Failed calls might indicate:
   - API rate limits
   - Network issues
   - Invalid API keys
   - Service outages

4. **Cost Analysis**: Different APIs have different pricing models. Faster responses might cost more per request.

### Example CSV Output
```csv
Timestamp,Prompt,Inception_Response,Inception_Time_Seconds,Inception_Success,OpenAI_Response,OpenAI_Time_Seconds,OpenAI_Success,Gemini_Response,Gemini_Time_Seconds,Gemini_Success
2024-01-15 14:30:22,"What are the benefits of renewable energy?","Renewable energy offers numerous benefits...",2.341,True,"Renewable energy sources like solar...",1.823,True,"The transition to renewable energy...",1.456,True
2024-01-15 14:30:22 - Timing Summary,Response Times (seconds),"Inception: 2.341s",,,"OpenAI: 1.823s",,,"Gemini: 1.456s",,
```

## Customization

- **Models**: Edit the model names in the script (e.g., change from `gpt-3.5-turbo` to `gpt-4`)
- **Parameters**: Adjust temperature, max_tokens, etc. in each API call function
- **Inception Labs Endpoint**: Update the `inception_endpoint` URL if using a different endpoint

## Error Handling

The tool handles various error scenarios:
- Missing API keys
- Network timeouts
- API rate limits
- Invalid responses

All errors are logged in the CSV output for analysis.
